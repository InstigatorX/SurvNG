# INT8 execution and native multistream investigation

Checked 2026-09-15 EDT on the deployed native-first branch. Batching and architecture changes are investigation only.

## Current execution path

The running process already uses `survng.dlstreamer_live --supervisor`: one process hosts the camera graphs. Its selected model is `models/e2e_int8_openvino_model/best.xml`, device GPU, with shared `model-instance-id=survng-best-GPU`, four inference requests and two execution streams. There are 13 camera streams, nine with detection enabled at the initial check.

Each camera retains its own source/decode/sampling/detection/tracker path, but its `gvadetect` shares the compiled model and inference resources with the other detection paths. This is the same resource-sharing mechanism documented by [DL Streamer](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/elements/gvadetect.html). Admin labels now describe requests/streams as shared, correcting the previous per-camera wording.

Explicit `batch-size` is 1. OpenVINO automatic batching is disabled because its wrapper previously failed to bind this pipeline's VA surface inputs. Shared requests, execution streams, explicit batching, and automatic batching are distinct controls; increasing one does not imply enabling the others.

## INT8 / DP4A evidence

- Hardware: Raptor Lake-P Intel UHD iGPU, PCI `8086:a7a8`, OpenVINO GPU architecture `v12.3.0`.
- Installed OpenVINO: `2026.2.0-21903-52ddc073857`; OpenCL driver: `26.31.39395.13`; DL Streamer: `2026.2.0`.
- Device capabilities advertise INT8 and integer dot products. Model IR contains 125 FakeQuantize operations, 94 Convolution operations and eight GroupConvolution operations.
- An isolated probe compiled the selected model on this GPU with the same throughput/stream/auto-batching settings, enabled profiling, and executed three inferences. All 102 executed convolution operations used `u8` kernels: 38 general IMAD, 55 IMAD 1×1, eight depthwise IMAD, and one MMAD input convolution.
- A second isolated compilation enabled Intel IGC shader dumps. **102 generated convolution machine-assembly files contain actual `dp4a` instructions.** For example, the generated `convolution_gpu_b_fs_zyx_fsv16_imad` kernel contains `dp4a (16|M0)` instructions. The dump/profiling flags were confined to the probe process and were not applied to the service.

This verifies the selected model's hardware INT8/DP4A execution path on this host. It is not an instruction trace attached to the running camera process: the probe uses input tensors rather than live VA surfaces. Floating-point preprocessing, GEMM, softmax and output operations remain; an INT8 model need not execute every operation as an integer. DP4A selection does not establish globally optimal FPS or latency.

Method: [OpenVINO GPU profiling](https://github.com/openvinotoolkit/openvino/blob/releases/2026/2/src/plugins/intel_gpu/docs/gpu_debug_utils.md), [Intel IGC shader dumps](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/shader_dumps_instruction.md), and [Intel instruction reference](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/6_instructions.md). Local diagnostic artifacts: `/tmp/survng-int8-profile.json`, `/tmp/survng-int8-runtime.xml`, `/tmp/survng-igc-probe/`.

## Batching options

| Option | Fit for SurvNG | Constraints |
| --- | --- | --- |
| Shared `model-instance-id`, small explicit batch | Smallest extension of the current runtime | Test batch 2 and 4, source fairness, startup, late frames and reconnects. Keep trackers camera-local. |
| `gvastreammux` → inference → demux → camera trackers | Explicit central frame scheduling | Requires source identity and timestamp preservation, caps compatibility and independent recovery. A mux alone does not guarantee a batched model invocation. |
| DL Streamer Pipeline Server | Packaged pipeline lifecycle/API and batching configuration | Adds service integration; uses the same underlying shared-model/batch controls. |
| SceneScape | Useful for calibrated scene coordinates and sensor/cross-camera fusion | Substantially larger scope than GPU scheduling; not a prerequisite for batching. |

[Pipeline Server batching documentation](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer-pipeline-server/advanced-guide/detailed_usage/how-to-advanced/cross-stream-batching.html) supports shared inference elements plus explicit batch sizes. Camera frames are grouped according to arrival; a batch can contain multiple frames from one camera. Batching may improve throughput, but it does not process whole camera frames in one GPU clock cycle, nor guarantee lower incident latency.

The installed plugin reports `batch-timeout` **unsupported with `va` and `va-surface-sharing` preprocessing**. That timeout belongs to OpenVINO automatic batching; it should not be assumed to make explicit VA batches safe under missing inputs. A proposed batch of 13 must not depend on all 13 cameras contributing: only nine currently detect, and cameras can disconnect. Small batches are the first experimental target, not an unconditional fleet-sized batch.

The selected model has input `[1,3,640,640]` and output `[1,300,6]`. An isolated OpenVINO reshape to batch 2 succeeds, producing `[2,300,6]`; reshape success alone does not prove native postprocessing, surface binding or output routing correctness.

An isolated native smoke test then ran two 640×640 NV12 VA-backed synthetic sources, with shared `gvadetect`, four requests, two execution streams and automatic batching disabled. **Batch 1 and explicit batch 2 both delivered all eight frames (four per source) and reached EOS.** The test initially stalled without downstream queues even at batch 1; adding the normal queue after each detector resolved that harness issue. This confirms basic native batch-2 initialization/frame flow with the loaded INT8 model, not a latency/throughput improvement, detection accuracy, camera fairness or outage recovery. No live graph was modified.

The installed `gvastreammux`/`gvastreamdemux` elements expose PTS synchronization, bounded queues and a 40 ms default maximum wait. In passthrough mode, mux inputs must have identical caps; these cameras have differing resolutions/aspect ratios. Container mode supports differing caps but cannot feed `gvadetect` directly. A shared detector interval greater than one after multiplexing can systematically skip particular cameras, so cadence must remain per-source. See [mux documentation](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/elements/gvastreammux.html).

## Framework / SceneScape recommendation

SurvNG already uses Python/GStreamer bindings to orchestrate C++ native media/inference elements in one process. Moving orchestration to C++ does not itself pool more GPU work; scheduling, batching, preprocessing and queue behavior determine that. Extend this supervisor before replacing it.

[SceneScape](https://docs.openedgeplatform.intel.com/dev/scenescape/index.html) adds a scene graph with spatial awareness and camera/sensor integration. Its Docker Compose video service uses DL Streamer Pipeline Server and supports batching configuration. Its documented Kubernetes deployment currently runs one camera pipeline per pod and explicitly lacks cross-stream batching. See [deployment-specific limitations](https://docs.openedgeplatform.intel.com/dev/scenescape/other-topics/how-to-configure-dlstreamer-video-pipeline.html).

Use standard GStreamer Python/C++ APIs for a future experiment. Do not start a rewrite around DL Streamer's preview Architecture 2.0 APIs: Intel documents that approach as being deprecated in favor of GStreamer analytics integration. [Architecture note](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-libraries/dlstreamer/architecture_2.0/architecture_2.0.html).

Recommended next experiment, not implemented: compare the existing shared batch-1 path against batches 2/4, testing `va` versus `va-surface-sharing` preprocessing. Intel's [performance guide](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/dev_guide/performance_guide.html) notes that the VA scaler can outperform surface-sharing resize on integrated GPUs. Measure per-camera fresh FPS, frame age and p95 latency, GPU/CPU utilization, fairness, startup and one-camera disconnect/reconnect. Preserve exact source/session/PTS associations for covers and replay. No speedup is claimed without those measurements.
