# isolated inference.py (stripped lerobot inference script)


"""
stripped LeRobot VLA inference script.

Data flow:
    LeRobotDataset(DATASET_PATH)        → ds_meta  (features + normalization stats)
    load_model(policy_path, ds_meta)    → policy, policy_cfg
    load_pipeline(policy_cfg, ds_meta)  → preprocessor, postprocessor
    run_inference(robot, policy, ...)   → sends actions to hardware
"""


# ─────────────────────────────────────────────
# IMPORTS:
# ─────────────────────────────────────────────

import time
import logging

# used for performance analysis
# (trying to check bottlenecks during inference)
import matplotlib.pyplot as plt

import torch
import numpy as np


# used for attention heatmaps:
import cv2
import imageio



# use for profling
from torch.profiler import profile, ProfilerActivity, record_function, tensorboard_trace_handler


from lerobot.configs.policies import PreTrainedConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    make_default_processors,
    RobotAction 
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

from pathlib import Path


# use for attn visualziation at end of inference
viz_frames = []


# ─────────────────────────────────────────────
# 0. CONFIG:
# ─────────────────────────────────────────────
"""
TODO: nickname dict for models would be useful
"""

# model things
POLICY_PATH  = "grahamwichhh/pi05_30k"
DATASET_PATH = "grahamwichhh/v5_pick-up-cube"   # training dataset, ONLY needed for feature shapes + norm stats
# NOTE (more info on why we need the dataset):
#   its extremely annoying but i guess metadata about joint angles/raw uint8 images -> whatever shape they need
#   to be for the model is stored in lerobots dataset object

TASK         = "Pick up the yellow cube."              # natural language prompt passed to the VLA model
DEVICE       = "cuda"                       # "cuda", "mps", or "cpu"
FPS          = 30                           # control loop frequency
RUN_TIME_S   = 120                          # how long to run inference (seconds)


# camera ports senmt to OpenCVCameraConfig require Path objects:
CAMERA_VIDEO_1 = Path("/dev/video0") # front view
CAMERA_VIDEO_2 = Path("/dev/video2") # 45 degree side view
CAMERA_VIDEO_3 = Path("/dev/video4") # wrist cam

ROBOT_CONFIG = so_follower.SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="rocky",                                          # must match th ASDDSAe id used during calibration
    calibration_dir=Path("~/.cache/huggingface/lerobot/calibration/robots/so_follower").expanduser(),
    cameras={
        "camera1": OpenCVCameraConfig(
            index_or_path=CAMERA_VIDEO_1,
            width=640,
            height=480,
            fps=30,
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path=CAMERA_VIDEO_2,
            width=640,
            height=480,
            fps=30,
        ),
        "camera3": OpenCVCameraConfig(
            index_or_path=CAMERA_VIDEO_3,
            width=640,
            height=480,
            fps=30,
        ),
    },
)



# things for timing plots:
# TODO: better naming scheme than "testing_" ... place after model name and send that to plot name?
TIMING_PLOT_NAME = f"testing_{time.strftime('%Y%m%d_%H%M%S')}"
timing_history = {
    "camera_capture": [],
    "obs_processing": [],
    "predict_action": [],
}
plt.ion()  # interactive mode - allows non-blocking updates
fig, ax = plt.subplots()
ax.set_xlabel("iteration")
ax.set_ylabel("ms")
ax.set_title(f"{TIMING_PLOT_NAME}")


# functionality to return so101 to resting position:
REST_POSITION = {
    "shoulder_pan.pos":  1.4,
    "shoulder_lift.pos": -99.0,
    "elbow_flex.pos":    97.0,
    "wrist_flex.pos":    72.0,
    "wrist_roll.pos":    -3.0,
    "gripper.pos":       3.2,
}


# below deals with XVLA annoying renaming scheme
USING_XVLA = False  # set to True if using a policy trained with the XVLA_RENAME_MAP feature renaming

XVLA_RENAME_MAP = {}

if USING_XVLA:
    logging.info("Using XVLA_RENAME_MAP for feature key remapping. This should match the rename map used during training.")
    XVLA_RENAME_MAP = {
        "observation.images.camera1": "observation.images.image",
        "observation.images.camera2": "observation.images.image2",
        "observation.images.camera3": "observation.images.image3",
    }


# optims NOTE: not deterministically better
USE_AUTOCAST = False
USE_AMP = False  # automatic mixed precision - can reduce latency, may cause instability on some models






# ─────────────────────────────────────────────
# 0.5. HELPER FUNCTIONS
# ─────────────────────────────────────────────

# defined helper function to end loop early if at rest position
def check_at_rest_position(obs, threshold=10.0):
    for joint, rest_value in REST_POSITION.items():
        if abs(obs[joint] - rest_value) > threshold:
            return False
    return True




# ─────────────────────────────────────────────
# 1. LOAD DATASET METADATA
# ─────────────────────────────────────────────
"""
    Fetch metadata from the training dataset without downloading any episode frames.
    LeRobotDataset(repo_id) pulls only the dataset card and stats JSON from HuggingFace.
    What we get back:
    ds_meta.features: shapes + dtypes the model expects (action, observations, etc.)
    ds_meta.stats: per-feature mean/std used for normalization during training
    Both are required downstream:
    make_policy()             needs ds_meta to validate input/output feature shapes
    make_pre_post_processors() needs ds_meta.stats to build the normalization layers
"""

# test loading dataset here:
def load_dataset(dataset_path: str, rename_map=None):

    print("rename map:")
    print(rename_map)

    logging.info(f"Loading dataset metadata from: {dataset_path}")
    # dataset = LeRobotDataset.create(repo_id=dataset_path, robot_type=ROBOT_NAME, fps=FPS, features=dataset_features)
    dataset=LeRobotDataset(dataset_path)
    logging.info(f"Dataset meta loaded:\n{dataset.meta}")


    """
    should be like:
        LeRobotDatasetMetadata({
            Repository ID: 'grahamwichhh/v3_so101-pick-up-lego',
            Total episodes: '50',
            Total frames: '21238',
            Features: '['action', 'observation.state', 'observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']',
        })',

    """

    # xvla uses different naming scheme. if using xvla we need to rename things
    if USING_XVLA:
        changed_features = []

        for feature in dataset.meta.features:
            if feature in rename_map:
                new_key = rename_map[feature]
                temp_feature_dict = {new_key: dict(dataset.meta.features[feature])}
                changed_features.append((feature, new_key))

        for old_key, new_key in changed_features:
            dataset.meta.features[new_key] = dataset.meta.features.pop(old_key)
            logging.info("Renamed feature '%s' -> '%s'", old_key, new_key)  

        print("renamed dataset features:")
        print(dataset.meta.features)


    return dataset


# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────


def load_model(policy_path: str, device: str, ds_meta):
    """
    Load the pretrained VLA policy.


    PreTrainedConfig.from_pretrained() reads the model architecture config
    (e.g. Pi0, ACT, Diffusion) stored alongside the weights on HuggingFace.


    make_policy() instantiates the correct model class, loads weights, and uses
    ds_meta to validate that the model's expected input/output features match
    the dataset the policy was trained on.


    ds_meta comes from load_dataset_meta() - it must be the dataset the policy
    was trained on, not an arbitrary dataset.
    """
    logging.info(f"Loading policy from: {policy_path}")





    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device


    # 
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    policy_cfg.rtc_config = RTCConfig()  # or with custom params


    # "Missing key(s) in state_dict" weight loading errors seen with ds_meta=None,
    # because make_policy uses ds_meta.features to correctly configure the model
    # head dimensions before loading weights.

    policy = make_policy(policy_cfg, ds_meta=ds_meta)

    # NOTE: prints layers and torch.Sizes but states many are empty
    # for name, param in policy.named_parameters():
    #     print(name, param.shape)

    # for name, module in policy.named_modules():
    #     if isinstance(module, type(module)) and not list(module.parameters(recurse=False)):
    #         print(f"No params: {name} ({type(module).__name__})")

    # alternate way to print policy
    # total = sum(p.numel() for p in policy.parameters())
    # total_buf = sum(b.numel() for b in policy.buffers())
    # print(f"Policy: {type(policy).__name__} | params={total:,} | buffers={total_buf:,}")

    # spot-check one weight that should exist
    # for name, buf in policy.named_buffers():
    #     print(name, buf.shape)
    #     break





    policy.eval()   # disable dropout etc. required for deterministic inference





    # ATTEMPTED OPTIMS

    # halve weights to fp16:
    # policy = policy.half()

    # policy = torch.compile(policy, mode="reduce-overhead")
    # mode options:
    #   "default"         - balanced
    #   "reduce-overhead" - best for repeated same-shape inputs (your case)
    #   "max-autotune"    - slowest to compile, fastest at runtime

    policy_cfg.use_amp = USE_AMP

    logging.info("Policy loaded successfully.")
    return policy, policy_cfg




# ─────────────────────────────────────────────
# 3. LOAD PIPELINE (pre/postprocessors)
# ─────────────────────────────────────────────


def load_pipeline(policy_cfg, ds_meta, rename_map=None):

    # NOTE: at this point in time, rename_map is not needed since the dataset features
    #       have been remapped in load_dataset()
    # TODO: double check this does not break anything in the preprocessing

    """
    Build the normalization pipelines that wrap the model.


    Preprocessor:  raw obs dict → normalized tensors the model expects
    Postprocessor: raw model output tensors → denormalized joint targets


    ds_meta.stats contains the per-feature mean/std saved at training time.
    rename_stats() applies RENAME_MAP to those stat keys so they match
    whatever observation key names your robot produces at runtime.


    Mirrors lerobot_record.py lines 484-492.
    """

    rename_map = rename_map if rename_map is not None else {}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),  # lerobot_record.py line 487
        preprocessor_overrides={
            "device_processor":              {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return preprocessor, postprocessor




# ─────────────────────────────────────────────
# 4. INFERENCE LOOP
# ─────────────────────────────────────────────
def run_inference(robot, policy, policy_cfg, preprocessor, postprocessor, task, fps, run_time_s, dataset):

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


    # TODO: this is redundant with renaming in load_dataset()
    if USING_XVLA:
        robot_to_policy_key_map = {
            "camera1": "observation.images.image",
            "camera2": "observation.images.image2",
            "camera3": "observation.images.image3",
        }
    else:
        robot_to_policy_key_map = {
            "camera1": "observation.images.camera1",
            "camera2": "observation.images.camera2",
            "camera3": "observation.images.camera3",
        }





    device = get_safe_torch_device(policy_cfg.device)
    logging.info(f"Starting inference loop | task='{task}' | fps={fps} | duration={run_time_s}s")

    start_t = time.perf_counter()
    timestamp = 0.0


    # velocity clamping (attempted smoothing)
    # prev_action: dict | None = None
    # max_joint_vel_deg_per_s = 125.0     # delta defines max movement PER TICK (can be multiple per second)
    # max_delta_deg = max_joint_vel_deg_per_s / FPS    # a possible result is moving 15 (units?) rather than 34 in a single second





    while timestamp < run_time_s:
        loop_start_t = time.perf_counter()

        # --- OBSERVE ---

        t0 = time.perf_counter() # start time

        raw_obs = robot.get_observation()
        t1 = time.perf_counter() # check how long it takes to get the robot state

        obs = robot_observation_processor(raw_obs)
        t2 = time.perf_counter() # check how long observation processing took


        # bypass build_dataset_frame (dataset issues)
        #   the observation batch can be tensors directly from policy_cfg.input_features.
        observation_frame = {}



        # joint state: robot outputs short keys like 'shoulder_pan.pos',
        # policy expects them aggregated under 'observation.state'
        state_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ]

        
        # observation_frame["observation.state"] = state_tensor.to(device) # TODO: purpose? use for triton?

        # state_tensor = torch.tensor(
        #     [obs[k] for k in state_keys], dtype=torch.float32
        # ).unsqueeze(0)  # shape: (1, 6) - batch dim required


        # Camera images - just remap the key, no conversion needed.
        # predict_action calls prepare_observation_for_inference internally,
        # which converts numpy arrays to tensors itself.
        for robot_key, policy_key in robot_to_policy_key_map.items():

            # BELOW IS TESTING TO OPTIMIZE IMAGE PREPROC Y SENDING TO GPU
            # if robot_key in obs and policy_key in policy_cfg.input_features:
            #     img_np = obs[robot_key]                           # (H,W,3) uint8 numpy
            #     buf = _pinned_buffers[policy_key]
            #     buf.copy_(torch.from_numpy(img_np))               # CPU pinned, one copy
            #     img_gpu = buf.cuda(non_blocking=True)             # async DMA, no bounce buffer
            #     img_gpu = img_gpu.permute(2,0,1).float().div_(255.0).unsqueeze(0)
            #     # (1, 3, H, W) float32 on GPU - preprocessor will skip re-transfer
            #     observation_frame[policy_key] = img_gpu


            if robot_key in obs and policy_key in policy_cfg.input_features:
                observation_frame[policy_key] = obs[robot_key]  # raw numpy array, HWC uint8

        
        # Joint state - same, just pass the numpy array
        observation_frame["observation.state"] = np.array(
            [obs[k] for k in state_keys], dtype=np.float32
        )

    

        # --- PREDICT ---
        """
            call predict_action(), will return a base tensor
                expects np.ndarray
                returns torch.Tensor

            example: 
                action_values are tensor([[  2.3541, -17.7406,  46.0053,  58.2234,  26.3015,  15.2118]])
        """




        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy_cfg.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )

        t3 = time.perf_counter() # check how long it takes to prediction the action





        # extract_attention_heatmaps lives in modeling_pi05 and reads the buffer
        # snapshot that sample_actions stored on the model after the denoising loop.
        # We pass the raw camera frames from obs so the overlay is on original resolution.
        # TODO: heatmaps toggle!!
        from lerobot.policies.pi05.modeling_pi05 import extract_attention_heatmaps

        attn_snapshot = getattr(policy.model, "last_attn_buffer_snapshot", None)
        if attn_snapshot:
            # Build raw_camera_frames list in the same order as the model's camera keys
            raw_frames = [
                obs[robot_key]                        # HWC uint8 numpy, original resolution
                for robot_key in ["camera1", "camera2", "camera3"]
                if robot_key in obs
            ]

            heatmaps = extract_attention_heatmaps(
                raw_camera_frames=raw_frames,
                attn_buffer=attn_snapshot,
            )

            if heatmaps is not None:
                composite = np.concatenate(heatmaps, axis=1)          # [H, W*3, 3] BGR
                viz_frames.append(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))  # imageio wants RGB






        # must convert it to "action_processed_policy" via make_robot_action using dataset as well
        """
            should be like: 
                {'shoulder_pan.pos': 2.354058265686035, 
                'shoulder_lift.pos': -17.740571975708008, 
                'elbow_flex.pos': 46.00529479980469, 
                'wrist_flex.pos': 58.223419189453125, 
                'wrist_roll.pos': 26.301464080810547, 
                'gripper.pos': 15.21180248260498}

        """

        action_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        # --- SEND ---
        robot_action_to_send = robot_action_processor((action_processed_policy, obs))

        robot.send_action(robot_action_to_send)

        # --- PACE TO FPS ---

        # loop_start_t ─────────────────────────────────────> now
        #       [observe → predict → send]  [sleep]
        #       |←────── dt_s ─────────────|←──────→|
        #       |←──────────── 1/fps (33.3ms) ──────→|

        dt_s = time.perf_counter() - loop_start_t
        sleep_time_s = 1.0 / fps - dt_s
        # if sleep_time_s < 0:
        #     logging.warning(f"Loop running slow: {1/dt_s:.1f} Hz vs target {fps} Hz")
        precise_sleep(max(sleep_time_s, 0.0))






        # matplot:
        timing_history["camera_capture"].append(1000 * (t1 - t0))
        timing_history["obs_processing"].append(1000 * (t2 - t1))
        timing_history["predict_action"].append(1000 * (t3 - t2))

        # remove initial outlier of model being sent to device:
        if len(timing_history["predict_action"]) == 1:
            # set to 0
            timing_history["predict_action"][0] = 0.0

        if len(timing_history["camera_capture"]) % 10 == 0:
            ax.clear()
            iterations = range(len(timing_history["camera_capture"]))
            ax.plot(iterations, timing_history["camera_capture"], label="camera_capture")
            ax.plot(iterations, timing_history["obs_processing"], label="obs_processing")
            ax.plot(iterations, timing_history["predict_action"], label="predict_action")
            ax.legend()
            ax.set_xlabel("iteration")
            ax.set_ylabel("ms")
            ax.set_title("per-iteration timing")
            plt.pause(0.001)  # non-blocking draw - 1ms, won't affect FPS meaningfully


        timestamp = time.perf_counter() - start_t
        

        # check if at rest position every n seconds
        # buffer after minimum seconds so dont end the moment loop starts:
        if timestamp > 20.0 and timestamp % 100.0 < 1.0:
            if check_at_rest_position(obs):
                logging.info("At resting position...")
                break


        # exit()








# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():

    # attempt to flush gpu memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()



    init_logging()


    # Step 1: fetch feature shapes + normalization stats from the training dataset
    # ds_meta = load_dataset_meta(DATASET_PATH)
    dataset=load_dataset(DATASET_PATH, rename_map=XVLA_RENAME_MAP)

    # Step 2: load policy weights, validated against ds_meta feature shapes
    policy, policy_cfg = load_model(POLICY_PATH, DEVICE, dataset.meta)

    # Step 3: build normalization pipelines using ds_meta.stats
    preprocessor, postprocessor = load_pipeline(policy_cfg, dataset.meta)


    # Step 4: connect robot and run
    # NOTE: robot calibration information is loaded (if present) here
    robot = make_robot_from_config(ROBOT_CONFIG)
    robot.connect()


    try:     
        if (USE_AUTOCAST):
            print("\n\nrunning inference with torch.autocast\n\n")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # run_inference(
                #     robot=robot,
                #     policy=policy,
                #     policy_cfg=policy_cfg,
                #     preprocessor=preprocessor,
                #     postprocessor=postprocessor,
                #     task=TASK,
                #     fps=FPS,
                #     run_time_s=RUN_TIME_S,
                #     dataset=dataset
                # )
                from torch.profiler import profile, ProfilerActivity, record_function, tensorboard_trace_handler

                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=False,
                    on_trace_ready=tensorboard_trace_handler(f"./torch_profiling/profiler"),
                ) as prof:
                    run_inference(
                        robot=robot,
                        policy=policy,
                        policy_cfg=policy_cfg,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        task=TASK,
                        fps=FPS,
                        run_time_s=RUN_TIME_S,
                        dataset=dataset
                    )
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                prof.export_chrome_trace("trace.json")


        else:
            print("\n\nrunning standard inference\n\n")


            run_inference(
                robot=robot,
                policy=policy,
                policy_cfg=policy_cfg,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=TASK,
                fps=FPS,
                run_time_s=RUN_TIME_S,
                dataset=dataset
            )



            # torch profiling NOTE: sometimes does not actually run inference when compiling cuda graphs? TODO: check
            # usage: uncomment, after saved run [tensorboard --logdir=./torch_profiling/profiler/]

            # from torch.profiler import profile, ProfilerActivity, record_function, tensorboard_trace_handler
            # with profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     record_shapes=True,
            #     with_stack=False,
            #     on_trace_ready=tensorboard_trace_handler(f"./torch_profiling/profiler"),
            # ) as prof:
            #     run_inference(
            #         robot=robot,
            #         policy=policy,
            #         policy_cfg=policy_cfg,
            #         preprocessor=preprocessor,
            #         postprocessor=postprocessor,
            #         task=TASK,
            #         fps=FPS,
            #         run_time_s=RUN_TIME_S,
            #         dataset=dataset
            #     )
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            # prof.export_chrome_trace("trace.json")







    finally:


        logging.info("Attempting to return to resting position.")
        rest_start_t = time.perf_counter()
        rest_duration_s = 5.0



        # calculate average predit_action time and log it:
        avg_predict_time = np.mean(timing_history["predict_action"])
        std_predict_time = np.std(timing_history["predict_action"])
        logging.info(f"Average predict_action time: {avg_predict_time:.1f} ms")
        logging.info(f"Std dev predict_action time: {std_predict_time:.1f} ms")


        # Add statistics text to plot
        ax.clear()
        iterations = range(len(timing_history["camera_capture"]))
        ax.plot(iterations, timing_history["camera_capture"], label="camera_capture")
        ax.plot(iterations, timing_history["obs_processing"], label="obs_processing")
        ax.plot(iterations, timing_history["predict_action"], label="predict_action")
        ax.legend(loc='upper left')
        ax.set_xlabel("iteration")
        ax.set_ylabel("ms")
        ax.set_title("per-iteration timing")
        
        # Add statistics as text on plot
        stats_text = f"predict_action avg: {avg_predict_time:.1f} ms\nstd dev: {std_predict_time:.1f} ms"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.ioff()
        plt.savefig(f"{'inference_timing_plots' + '/' + TIMING_PLOT_NAME}.png")
        plt.close()

        print("closing plot")


        # read curr pos:
        current_obs = robot.get_observation()
        current_pos = {k: current_obs[k] for k in REST_POSITION}

        while time.perf_counter() - rest_start_t < rest_duration_s:
            # linearly interpolate from current position to rest over rest_duration_s
            # alpha goes 0.0 → 1.0 over the duration
            # (claude code, TODO: learn about interpolation)
            alpha = (time.perf_counter() - rest_start_t) / rest_duration_s
            alpha = min(alpha, 1.0)

            interpolated = {
                k: current_pos[k] + alpha * (REST_POSITION[k] - current_pos[k])
                for k in REST_POSITION
            }

            # print(interpolated)

            robot.send_action(interpolated)
            precise_sleep(1.0 / FPS)


        # heatmap visualization occurs last (sometimes heatmap saving skips resting pos action)
        if viz_frames:
            video_path = f"attention_videos/attn_vis_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            imageio.mimwrite(video_path, viz_frames, fps=FPS, codec="libx264")
            logging.info(f"Saved attention visualization to {video_path}")
        else:
            print("did not make video, viz_frames DNE")


        # Always disconnect cleanly even if an exception is raised mid-loop
        robot.disconnect()
        logging.info("Done.")




if __name__ == "__main__":
    main()
