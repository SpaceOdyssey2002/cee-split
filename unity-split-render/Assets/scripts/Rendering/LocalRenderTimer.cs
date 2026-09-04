using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class LocalRenderTimer : MonoBehaviour
{
    public enum GpuTimingSource
    {
        None,
        CameraRecorder,
        WholeFrameFallback
    }

    [Header("GPU Timing")]
    public bool measureGpuTime = true;

    [Tooltip("Android Vulkan cannot expose per-camera GPU samples. Use whole-frame GPU time when that happens.")]
    public bool allowWholeFrameFallback = true;

    [Range(0.01f, 1f)]
    public float gpuSmoothing = 0.2f;

    [Min(0.5f)]
    public float recorderFallbackDelaySeconds = 2f;

    private readonly Stopwatch cpuStopwatch = new Stopwatch();
    private readonly FrameTiming[] frameTimings = new FrameTiming[4];
    private readonly List<string> samplerNames = new List<string>(256);

    private Camera targetCamera;
    private SenderFramePacker layerPacker;
    private double cpuSubmissionTimeMs;
    private double gpuRenderTimeMs;
    private bool hasGpuSample;
    private int lastRenderedFrame = -1;
    private float recorderWaitStartTime;
    private float nextRecorderDiscoveryTime;
    private int cpuAccumulationFrame = -1;

    private Recorder gpuRecorder;
    private string gpuSamplerName = "";
    private GpuTimingSource timingSource = GpuTimingSource.None;

    private void Awake()
    {
        targetCamera = GetComponent<Camera>();
        layerPacker = GetComponent<SenderFramePacker>();
    }

    private void OnEnable()
    {
        if (targetCamera == null)
            targetCamera = GetComponent<Camera>();

        recorderWaitStartTime = Time.realtimeSinceStartup;
        DiscoverCameraRecorder();
        RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
    }

    private void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;

        if (gpuRecorder != null)
            gpuRecorder.enabled = false;
    }

    private void Update()
    {
        if (!measureGpuTime || !IsCameraActivelyRendering())
            return;

        // Per-layer rendering uses several cameras. Whole-frame GPU timing is
        // the supported aggregate sample for that path.
        if (layerPacker != null && layerPacker.HasActiveLayerCameras)
        {
            TryReadWholeFrameGpuTime();
            return;
        }

        if (TryReadCameraRecorder())
            return;

        if (Time.realtimeSinceStartup >= nextRecorderDiscoveryTime)
        {
            DiscoverCameraRecorder();
            nextRecorderDiscoveryTime = Time.realtimeSinceStartup + 1f;
        }

        bool recorderTimedOut = Time.realtimeSinceStartup - recorderWaitStartTime >=
                                Mathf.Max(0.5f, recorderFallbackDelaySeconds);
        if (allowWholeFrameFallback && recorderTimedOut)
            TryReadWholeFrameGpuTime();
    }

    private bool TryReadCameraRecorder()
    {
        if (!SystemInfo.supportsGpuRecorder || gpuRecorder == null || !gpuRecorder.isValid)
            return false;

        if (!gpuRecorder.enabled)
            gpuRecorder.enabled = true;

        if (gpuRecorder.gpuSampleBlockCount <= 0)
            return false;

        double sampleMs = gpuRecorder.gpuElapsedNanoseconds / 1_000_000.0;
        if (sampleMs <= 0.0)
            return false;

        SetGpuSample(sampleMs, GpuTimingSource.CameraRecorder);
        return true;
    }

    private void DiscoverCameraRecorder()
    {
        if (!measureGpuTime || !SystemInfo.supportsGpuRecorder || targetCamera == null)
            return;

        string expectedName = $"UniversalRenderPipeline.RenderSingleCameraInternal: {targetCamera.name}";
        Recorder expectedRecorder = Recorder.Get(expectedName);
        if (expectedRecorder != null && expectedRecorder.isValid)
        {
            SetRecorder(expectedRecorder, expectedName);
            return;
        }

        samplerNames.Clear();
        Sampler.GetNames(samplerNames);
        for (int i = 0; i < samplerNames.Count; i++)
        {
            string candidate = samplerNames[i];
            if (!candidate.Contains("RenderSingleCameraInternal") ||
                !candidate.EndsWith(targetCamera.name))
                continue;

            Recorder candidateRecorder = Recorder.Get(candidate);
            if (candidateRecorder != null && candidateRecorder.isValid)
            {
                SetRecorder(candidateRecorder, candidate);
                return;
            }
        }
    }

    private void SetRecorder(Recorder recorder, string samplerName)
    {
        if (gpuRecorder != null && gpuRecorder != recorder)
            gpuRecorder.enabled = false;

        gpuRecorder = recorder;
        gpuSamplerName = samplerName;
        gpuRecorder.enabled = true;
    }

    private void TryReadWholeFrameGpuTime()
    {
        if (!FrameTimingManager.IsFeatureEnabled())
            return;

        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings((uint)frameTimings.Length, frameTimings);
        if (count == 0)
            return;

        double totalMs = 0.0;
        int validCount = 0;
        for (int i = 0; i < count; i++)
        {
            double sampleMs = frameTimings[i].gpuFrameTime;
            if (sampleMs <= 0.0)
                continue;

            totalMs += sampleMs;
            validCount++;
        }

        if (validCount > 0)
            SetGpuSample(totalMs / validCount, GpuTimingSource.WholeFrameFallback);
    }

    private void SetGpuSample(double sampleMs, GpuTimingSource source)
    {
        if (!hasGpuSample || timingSource != source)
            gpuRenderTimeMs = sampleMs;
        else
            gpuRenderTimeMs += (sampleMs - gpuRenderTimeMs) * Mathf.Clamp01(gpuSmoothing);

        hasGpuSample = true;
        timingSource = source;
    }

    private void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        if (!IsMonitoredCamera(camera))
            return;

        if (cpuAccumulationFrame != Time.frameCount)
        {
            cpuAccumulationFrame = Time.frameCount;
            cpuSubmissionTimeMs = 0.0;
        }
        lastRenderedFrame = Time.frameCount;
        cpuStopwatch.Restart();
    }

    private void OnEndCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        if (!IsMonitoredCamera(camera))
            return;

        cpuStopwatch.Stop();
        cpuSubmissionTimeMs += cpuStopwatch.Elapsed.TotalMilliseconds;
    }

    private void OnPreRender()
    {
        if (GraphicsSettings.currentRenderPipeline == null)
            cpuStopwatch.Restart();
    }

    private void OnPostRender()
    {
        if (GraphicsSettings.currentRenderPipeline != null)
            return;

        cpuStopwatch.Stop();
        cpuSubmissionTimeMs = cpuStopwatch.Elapsed.TotalMilliseconds;
    }

    public bool HasGpuTimingSample() => hasGpuSample;
    public bool IsGpuTimingSupported() =>
        SystemInfo.supportsGpuRecorder || FrameTimingManager.IsFeatureEnabled();
    public bool IsCameraActivelyRendering() =>
        (layerPacker != null && layerPacker.HasActiveLayerCameras) ||
        (targetCamera != null && targetCamera.isActiveAndEnabled && targetCamera.cullingMask != 0);
    public bool RenderedThisFrame() => lastRenderedFrame == Time.frameCount;
    public bool IsPerCameraGpuTiming() => timingSource == GpuTimingSource.CameraRecorder;
    public string GetGpuTimingSourceName()
    {
        if (timingSource == GpuTimingSource.CameraRecorder) return "camera-recorder";
        if (timingSource == GpuTimingSource.WholeFrameFallback) return "whole-frame-fallback";
        return "none";
    }

    public string GetGpuSamplerName() => gpuSamplerName;
    public double GetGpuRenderTimeMs() =>
        IsCameraActivelyRendering() && hasGpuSample ? gpuRenderTimeMs : 0.0;
    public double GetLastValidGpuRenderTimeMs() => hasGpuSample ? gpuRenderTimeMs : 0.0;
    public double GetCpuSubmissionTimeMs() => cpuSubmissionTimeMs;
    public double GetRenderTimeMs() => GetGpuRenderTimeMs();

    private bool IsMonitoredCamera(Camera camera)
    {
        return camera == targetCamera ||
            (layerPacker != null && layerPacker.OwnsLayerCamera(camera));
    }
}
