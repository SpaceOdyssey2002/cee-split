using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

// Attach this component to the rendering camera, move/rotate the camera through
// representative views, and make a standalone development build. The logger
// records only the three variables needed by the paper's effective-FLOP model.
[RequireComponent(typeof(Camera))]
public class RenderCalibrationLogger : MonoBehaviour
{
    [Header("Calibration")]
    public string nodeName = "End";
    public int warmupFrames = 30;
    public int samplesPerResolution = 300;
    public int baseWidth = 1920;
    public int baseHeight = 1080;
    public float[] resolutionScales =
    {
        0.3333333f,
        0.6666667f,
        1.0f,
        1.3333333f,
        2.0f
    };

    [Header("Automatic Camera Motion")]
    public bool enableAutomaticCameraMotion = true;
    [Min(0.0f)] public float horizontalRadius = 1.5f;
    [Min(0.0f)] public float forwardRadius = 1.0f;
    [Min(0.0f)] public float verticalAmplitude = 0.35f;
    [Min(0.0f)] public float yawAmplitudeDegrees = 15.0f;
    [Min(0.0f)] public float pitchAmplitudeDegrees = 5.0f;
    [Min(1.0f)] public float motionCycleSeconds = 30.0f;
    [Min(1.0f)] public float nominalFrameRate = 60.0f;

    [Header("Frame Timing")]
    [Tooltip("FrameTiming results arrive several frames late. These frames are captured and discarded after each camera move.")]
    [Min(0)] public int timingDelayFrames = 4;
    [Tooltip("Number of complete FrameTiming samples whose median is written as one CSV row.")]
    [Min(1)] public int timingFramesPerSample = 5;
    [Tooltip("Maximum attempts per requested row before the logger gives up on a resolution.")]
    [Min(1)] public int maxAttemptsPerSample = 3;

    [Header("SRP Batcher A/B Passes")]
    public bool captureWithSrpBatcherEnabled = true;
    public bool captureWithSrpBatcherDisabled = true;

    [Header("Output")]
    [Tooltip("Leave empty to use the active Unity scene name.")]
    public string sceneNameOverride = "";
    public string srpEnabledOutputFile = "unity_render_calibration_srp_on.csv";
    public string srpDisabledOutputFile = "unity_render_calibration_srp_off.csv";

    private Camera calibrationCamera;
    private StreamWriter writer;
    private RenderTexture originalTargetTexture;
    private Vector3 initialCameraPosition;
    private Quaternion initialCameraRotation;
    private bool originalSrpBatcherState;
    private bool runtimeStateCaptured;

    private IEnumerator Start()
    {
        calibrationCamera = GetComponent<Camera>();
        originalTargetTexture = calibrationCamera.targetTexture;
        initialCameraPosition = transform.position;
        initialCameraRotation = transform.rotation;
        originalSrpBatcherState = GraphicsSettings.useScriptableRenderPipelineBatching;
        runtimeStateCaptured = true;

        if (!captureWithSrpBatcherEnabled && !captureWithSrpBatcherDisabled)
        {
            Debug.LogWarning("Render calibration skipped because both SRP Batcher passes are disabled.");
            yield break;
        }

        if (captureWithSrpBatcherEnabled)
            yield return RunCalibrationPass(true, srpEnabledOutputFile);

        if (captureWithSrpBatcherDisabled)
            yield return RunCalibrationPass(false, srpDisabledOutputFile);

        RestoreRuntimeState();
        Debug.Log("Render calibration complete. Output directory: " + Application.persistentDataPath);
    }

    private IEnumerator RunCalibrationPass(bool enableSrpBatcher, string outputFile)
    {
        if (string.IsNullOrWhiteSpace(outputFile))
        {
            Debug.LogError("Render calibration output filename is empty; skipping this pass.");
            yield break;
        }

        GraphicsSettings.useScriptableRenderPipelineBatching = enableSrpBatcher;
        // Let the render pipeline observe the runtime state change before warm-up.
        yield return null;
        yield return null;

        string path = Path.Combine(Application.persistentDataPath, outputFile);
        writer = new StreamWriter(path, false);
        writer.WriteLine(
            "node,scene,srp_batcher,triangles,pixels,render_ms," +
            "cpu_frame_ms,cpu_main_thread_ms,cpu_render_thread_ms," +
            "cpu_present_wait_ms,gpu_ms");

        Debug.Log(
            "Starting render calibration with SRP Batcher " +
            (enableSrpBatcher ? "enabled" : "disabled") + ": " + path);

        foreach (float scale in resolutionScales)
        {
            int width = Mathf.Max(1, Mathf.RoundToInt(baseWidth * scale));
            int height = Mathf.Max(1, Mathf.RoundToInt(baseHeight * scale));
            RenderTexture target = new RenderTexture(width, height, 24);
            target.Create();
            calibrationCamera.targetTexture = target;

            ApplyAutomaticCameraPose(0.0f);

            for (int i = 0; i < warmupFrames; i++)
                yield return null;

            int validSamples = 0;
            int attempts = 0;
            int maximumAttempts = Mathf.Max(
                samplesPerResolution,
                samplesPerResolution * Mathf.Max(1, maxAttemptsPerSample));

            while (validSamples < samplesPerResolution && attempts < maximumAttempts)
            {
                attempts++;
                float elapsedSeconds =
                    validSamples *
                    (Mathf.Max(0, timingDelayFrames) + Mathf.Max(1, timingFramesPerSample)) /
                    Mathf.Max(1.0f, nominalFrameRate);
                ApplyAutomaticCameraPose(elapsedSeconds);

                // Hold the new camera pose while delayed results from the
                // previous pose drain from FrameTimingManager.
                for (int delayFrame = 0;
                     delayFrame < Mathf.Max(0, timingDelayFrames);
                     delayFrame++)
                {
                    FrameTimingManager.CaptureFrameTimings();
                    yield return new WaitForEndOfFrame();
                }

                List<double> renderCriticalPathSamples = new List<double>();
                List<double> cpuFrameSamples = new List<double>();
                List<double> cpuMainSamples = new List<double>();
                List<double> cpuRenderSamples = new List<double>();
                List<double> cpuPresentWaitSamples = new List<double>();
                List<double> gpuSamples = new List<double>();

                for (int timingFrame = 0;
                     timingFrame < Mathf.Max(1, timingFramesPerSample);
                     timingFrame++)
                {
                    FrameTimingManager.CaptureFrameTimings();
                    yield return new WaitForEndOfFrame();

                    FrameTiming[] latest = new FrameTiming[1];
                    uint count = FrameTimingManager.GetLatestTimings(1, latest);
                    if (count == 0)
                        continue;

                    double cpuFrame = NonnegativeFinite(latest[0].cpuFrameTime);
                    double cpuMain = NonnegativeFinite(latest[0].cpuMainThreadFrameTime);
                    double cpuRender = NonnegativeFinite(latest[0].cpuRenderThreadFrameTime);
                    double cpuPresentWait = NonnegativeFinite(
                        latest[0].cpuMainThreadPresentWaitTime);
                    double gpu = NonnegativeFinite(latest[0].gpuFrameTime);
                    double renderCriticalPath = Math.Max(cpuMain, Math.Max(cpuRender, gpu));

                    if (renderCriticalPath <= 0.0)
                        continue;

                    renderCriticalPathSamples.Add(renderCriticalPath);
                    cpuFrameSamples.Add(cpuFrame);
                    cpuMainSamples.Add(cpuMain);
                    cpuRenderSamples.Add(cpuRender);
                    cpuPresentWaitSamples.Add(cpuPresentWait);
                    gpuSamples.Add(gpu);
                }

                if (renderCriticalPathSamples.Count == 0)
                    continue;

                long triangles = CountVisibleTriangles(calibrationCamera);
                long pixels = (long)width * height;
                string sceneName = string.IsNullOrWhiteSpace(sceneNameOverride)
                    ? SceneManager.GetActiveScene().name
                    : sceneNameOverride.Trim();
                writer.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "{0},{1},{2},{3},{4},{5:F6},{6:F6},{7:F6},{8:F6},{9:F6},{10:F6}",
                    EscapeCsv(nodeName),
                    EscapeCsv(sceneName),
                    enableSrpBatcher ? "on" : "off",
                    triangles,
                    pixels,
                    Median(renderCriticalPathSamples),
                    Median(cpuFrameSamples),
                    Median(cpuMainSamples),
                    Median(cpuRenderSamples),
                    Median(cpuPresentWaitSamples),
                    Median(gpuSamples)));
                validSamples++;
            }

            if (validSamples < samplesPerResolution)
            {
                Debug.LogWarning(string.Format(
                    CultureInfo.InvariantCulture,
                    "Only {0}/{1} valid render samples were captured at {2}x{3} " +
                    "with SRP Batcher {4} after {5} attempts.",
                    validSamples,
                    samplesPerResolution,
                    width,
                    height,
                    enableSrpBatcher ? "enabled" : "disabled",
                    attempts));
            }

            writer.Flush();
            calibrationCamera.targetTexture = null;
            target.Release();
            Destroy(target);
        }

        writer.Close();
        writer = null;
        transform.SetPositionAndRotation(initialCameraPosition, initialCameraRotation);
        Debug.Log("Render calibration data saved to: " + path);
    }

    private static double NonnegativeFinite(double value)
    {
        return double.IsNaN(value) || double.IsInfinity(value) || value < 0.0
            ? 0.0
            : value;
    }

    private static double Median(List<double> values)
    {
        if (values == null || values.Count == 0)
            return 0.0;

        double[] sorted = values.ToArray();
        Array.Sort(sorted);
        int middle = sorted.Length / 2;
        return sorted.Length % 2 == 0
            ? (sorted[middle - 1] + sorted[middle]) * 0.5
            : sorted[middle];
    }

    private static string EscapeCsv(string value)
    {
        string safe = value ?? string.Empty;
        return "\"" + safe.Replace("\"", "\"\"") + "\"";
    }

    private void ApplyAutomaticCameraPose(float elapsedSeconds)
    {
        if (!enableAutomaticCameraMotion)
        {
            transform.SetPositionAndRotation(initialCameraPosition, initialCameraRotation);
            return;
        }

        float cycle = Mathf.Max(1.0f, motionCycleSeconds);
        float phase = 2.0f * Mathf.PI * Mathf.Repeat(elapsedSeconds / cycle, 1.0f);

        // A smooth deterministic three-dimensional figure-eight path. Every
        // resolution and both SRP passes replay exactly the same camera poses.
        Vector3 localOffset = new Vector3(
            horizontalRadius * Mathf.Sin(phase),
            verticalAmplitude * Mathf.Sin(3.0f * phase),
            forwardRadius * Mathf.Sin(2.0f * phase));
        transform.position = initialCameraPosition + initialCameraRotation * localOffset;

        float yaw = yawAmplitudeDegrees * Mathf.Sin(phase);
        float pitch = pitchAmplitudeDegrees * Mathf.Sin(2.0f * phase);
        transform.rotation = initialCameraRotation * Quaternion.Euler(pitch, yaw, 0.0f);
    }

    private static long CountVisibleTriangles(Camera cameraToTest)
    {
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(cameraToTest);
        Renderer[] renderers = FindObjectsByType<Renderer>(FindObjectsInactive.Exclude);
        long total = 0;

        foreach (Renderer renderer in renderers)
        {
            if (!renderer.enabled || !renderer.gameObject.activeInHierarchy ||
                !GeometryUtility.TestPlanesAABB(planes, renderer.bounds))
                continue;

            Mesh mesh = null;
            if (renderer is SkinnedMeshRenderer skinned)
                mesh = skinned.sharedMesh;
            else
            {
                MeshFilter filter = renderer.GetComponent<MeshFilter>();
                if (filter != null)
                    mesh = filter.sharedMesh;
            }

            if (mesh == null)
                continue;

            for (int submesh = 0; submesh < mesh.subMeshCount; submesh++)
                total += (long)mesh.GetIndexCount(submesh) / 3L;
        }

        return total;
    }

    private void OnDestroy()
    {
        if (writer != null)
        {
            writer.Close();
            writer = null;
        }

        RestoreRuntimeState();
    }

    private void RestoreRuntimeState()
    {
        if (!runtimeStateCaptured)
            return;

        if (calibrationCamera != null)
            calibrationCamera.targetTexture = originalTargetTexture;

        transform.SetPositionAndRotation(initialCameraPosition, initialCameraRotation);
        GraphicsSettings.useScriptableRenderPipelineBatching = originalSrpBatcherState;
    }
}
