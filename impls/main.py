import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
from absl import app, flags
# import wandb
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
#from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict  # , get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 200, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 50, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


import os
import flax.serialization

def save_agent(agent, save_dir, step):
    """Save the agent’s state to a file."""
    os.makedirs(save_dir, exist_ok=True)
    # Create a state dictionary. You can add more fields as needed.
    checkpoint_state = {
        'agent': agent,   # agent state (parameters, optimizer state, RNG, etc.)
        'step': step,
    }
    ckpt_path = os.path.join(save_dir, f"checkpoint_{step}.ckpt")
    with open(ckpt_path, "wb") as f:
        f.write(flax.serialization.to_bytes(checkpoint_state))
    print(f"Checkpoint saved at {ckpt_path}")

def restore_agent(agent, restore_dir, restore_step):
    """Restore the agent’s state from a file."""
    ckpt_path = os.path.join(restore_dir, f"checkpoint_{restore_step}.ckpt")
    with open(ckpt_path, "rb") as f:
        ckpt_bytes = f.read()
    # The second argument is your "empty" agent with the same structure.
    restored_state = flax.serialization.from_bytes({}, ckpt_bytes)
    # Here we assume that you want to update your agent with the saved state.
    # How you merge restored_state['agent'] into your current agent depends on your design.
    agent = restored_state['agent']
    print(f"Checkpoint restored from {ckpt_path}")
    return agent


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    # setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config['frame_stack']
    )

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent if specified.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        print(FLAGS.restore_path, FLAGS.restore_epoch)
        # If you saved at, say, step 10000, you can restore by:
        #agent = restore_agent(agent, FLAGS.save_dir, 10000)

    # Create CSV loggers.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))

    first_time = time.time()
    last_time = time.time()

    # ------------------------
    # Main training loop
    # ------------------------
    for i in range(1, FLAGS.train_steps + 1):
        # Sample a batch and update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # ------------------------------------------------------------------
        # Get the batch training loss using total_loss
        #   - This returns (loss, info) where 'loss' is the sum of actor,
        #     critic, and value losses. 'info' contains each component.
        # ------------------------------------------------------------------
        train_loss, train_loss_info = agent.total_loss(batch, grad_params=None)
        # Insert them into update_info for logging
        update_info['loss/total'] = float(train_loss)
        for k, v in train_loss_info.items():
            # e.g. actor/actor_loss, critic/critic_loss, value/value_loss, etc.
            update_info[f'loss/{k}'] = float(v)

        # Print loss every 10 batches
        if i % 10 == 0:
            print(f"Batch {i}: total_loss = {train_loss:.4f}")

        # Log metrics every FLAGS.log_interval steps.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            # Validation
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                val_loss, val_info = agent.total_loss(val_batch, grad_params=None)
                # You can log the validation total loss similarly:
                train_metrics['validation/loss/total'] = float(val_loss)
                for k, v in val_info.items():
                    train_metrics[f'validation/loss/{k}'] = float(v)

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()

            # wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

            save_agent(agent, FLAGS.save_dir, i)

        # Evaluate agent every FLAGS.eval_interval steps.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent

            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)

            # If your environment does not have task_infos, you may need
            # a fallback or skip multi-task handling:
            #task_infos = getattr(env.unwrapped, 'task_infos', [{'task_name': 'default_task'}])
            # task_infos = getattr(env.unwrapped, 'task_infos', env.task_infos)
            try:
                task_infos = env.unwrapped.task_infos
            except AttributeError:
                task_infos = [{'task_name': 'default_task'}]

            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)

            # For a quieter loop, we disable tqdm
            for task_id in range(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)

            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            # if FLAGS.video_episodes > 0:
            #     video = get_wandb_video(renders=renders, n_cols=num_tasks)
            #     eval_metrics['video'] = video

            # wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent every FLAGS.save_interval steps.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
            print(f"Agent saved at step {i} -> {FLAGS.save_dir}")

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
