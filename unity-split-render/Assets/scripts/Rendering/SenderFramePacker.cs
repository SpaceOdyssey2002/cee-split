using System;
using Unity.RenderStreaming;
using UnityEngine;
using UnityEngine.Rendering.Universal;

[DefaultExecutionOrder(-100)]
public class SenderFramePacker : MonoBehaviour
{
    [Header("Render Streaming")]
    public VideoStreamSender videoStreamSender;

    [Header("Color Resolution")]
    public int streamWidth = 1920;
    public int streamHeight = 1080;

    [Header("Per-layer rendering and shared stream")]
    [Tooltip("Maximum color resolution for the single-track RGB-D prototype. Unity.WebRTC limits one video track to 4K pixel count, so a vertically packed color+depth frame is capped at 1920x2160.")]
    public Vector2Int maximumColorResolution = new Vector2Int(1920, 1080);
    [Tooltip("Per-layer rendering resolution used before the first scheduler decision arrives.")]
    public Vector2Int initialColorResolution = new Vector2Int(1920, 1080);

    [Tooltip("Fixed shared packed texture. Its height must be twice the color height (RGB + depth).")]
    public RenderTexture packedRT;
    [HideInInspector] public Material packMaterial;

    private sealed class LayerSlot
    {
        public string name;
        public int layer;
        public bool rendersSky;
        public GameObject cameraObject;
        public Camera camera;
        public LayerFrameCapture capture;
        public RenderTexture colorRT;
        public RenderTexture packedLayerRT;
        public Vector2Int resolution;
        public bool active;
    }

    private RenderTexture runtimePackedRT;
    private Camera renderCamera;
    private Material layerCompositeMaterial;
    private LayerSlot nearSlot;
    private LayerSlot midSlot;
    private LayerSlot farSlot;
    private int initialCullingMask;

    private static readonly Vector2Int SafeMaximumColorResolution =
        new Vector2Int(1920, 1080);

    public Vector2Int MaximumColorResolution => maximumColorResolution;
    public Vector2Int CurrentColorResolution { get; private set; }
    public bool HasActiveLayerCameras =>
        (nearSlot != null && nearSlot.active) ||
        (midSlot != null && midSlot.active) ||
        (farSlot != null && farSlot.active);

    private void Awake()
    {
        renderCamera = GetComponent<Camera>();
        initialCullingMask = renderCamera != null ? renderCamera.cullingMask : 0;

        ClampToSingleTrackLimit();

        Shader packShader = Shader.Find("Custom/PackRgbDepth");
        if (packShader != null)
            packMaterial = new Material(packShader);
        else
            Debug.LogError("[SenderFramePacker] Custom/PackRgbDepth shader was not found.");

        Shader compositeShader = Shader.Find("Custom/CompositeLayerPacked");
        if (compositeShader != null)
            layerCompositeMaterial = new Material(compositeShader);
        else
            Debug.LogError("[SenderFramePacker] Custom/CompositeLayerPacked shader was not found.");

        EnsureSafePackedTexture();
        CreateLayerCameras();
    }

    private void Start()
    {
        if (renderCamera != null)
        {
            renderCamera.depthTextureMode |= DepthTextureMode.Depth;
            renderCamera.enabled = false;
        }

        bool nearActive = IncludesLayer(initialCullingMask, nearSlot);
        bool midActive = IncludesLayer(initialCullingMask, midSlot);
        bool farActive = IncludesLayer(initialCullingMask, farSlot);
        if (!nearActive && !midActive && !farActive)
            nearActive = midActive = farActive = true;

        ConfigureLayers(
            nearActive, initialColorResolution,
            midActive, initialColorResolution,
            farActive, initialColorResolution);

        if (packedRT == null)
        {
            Debug.LogError("[SenderFramePacker] packedRT is not assigned.");
            return;
        }

        // The WebRTC source stays at a stable maximum size. The scheduler changes
        // encoder downscaling while this component renders each layer separately.
        if (videoStreamSender != null)
        {
            if (videoStreamSender.sourceTexture != packedRT)
                videoStreamSender.sourceTexture = packedRT;

            videoStreamSender.SetTextureSize(new Vector2Int(packedRT.width, packedRT.height));
        }
    }

    private void LateUpdate()
    {
        SyncLayerCamera(nearSlot, 0);
        SyncLayerCamera(midSlot, 1);
        SyncLayerCamera(farSlot, 2);

        // Layer packing is produced by URP later in this frame. Compositing here
        // therefore consumes the previous completed layer textures, which is a
        // deliberate one-frame pipeline and avoids issuing Graphics.Blit from an
        // SRP end-frame callback (unreliable on Android and with multiple cameras).
        CompositeSharedPackedTexture();
    }

    private void ClampToSingleTrackLimit()
    {
        if (maximumColorResolution.x > SafeMaximumColorResolution.x ||
            maximumColorResolution.y > SafeMaximumColorResolution.y)
        {
            Debug.LogWarning(
                $"[SenderFramePacker] Requested maximum color resolution " +
                $"{maximumColorResolution.x}x{maximumColorResolution.y} would require a " +
                $"{maximumColorResolution.x}x{maximumColorResolution.y * 2} packed track. " +
                $"Unity.WebRTC rejects packed frames above 4K pixel count; clamping this " +
                $"single-track RGB-D prototype to {SafeMaximumColorResolution.x}x" +
                $"{SafeMaximumColorResolution.y} color ({SafeMaximumColorResolution.x}x" +
                $"{SafeMaximumColorResolution.y * 2} packed). ");
            maximumColorResolution = SafeMaximumColorResolution;
        }

        initialColorResolution = ClampResolution(initialColorResolution);
    }

    private void EnsureSafePackedTexture()
    {
        int requiredWidth = Mathf.Max(2, maximumColorResolution.x);
        int requiredHeight = Mathf.Max(2, maximumColorResolution.y) * 2;
        if (requiredWidth > SystemInfo.maxTextureSize || requiredHeight > SystemInfo.maxTextureSize)
        {
            Debug.LogError(
                $"[SenderFramePacker] Requested packed source {requiredWidth}x{requiredHeight} " +
                $"exceeds maxTextureSize={SystemInfo.maxTextureSize}.");
            return;
        }

        if (packedRT != null &&
            packedRT.width == requiredWidth && packedRT.height == requiredHeight)
        {
            packedRT.wrapMode = TextureWrapMode.Clamp;
            packedRT.filterMode = FilterMode.Bilinear;
            return;
        }

        runtimePackedRT = new RenderTexture(
            requiredWidth,
            requiredHeight,
            0,
            RenderTextureFormat.ARGB32)
        {
            name = $"PackedRGBD_Runtime_{requiredWidth}x{requiredHeight}",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
            autoGenerateMips = false
        };
        runtimePackedRT.Create();
        packedRT = runtimePackedRT;
        Debug.Log($"[SenderFramePacker] Created shared packed source {requiredWidth}x{requiredHeight}.");
    }

    private void CreateLayerCameras()
    {
        nearSlot = CreateLayerCamera("Near", "RenderNear", false);
        midSlot = CreateLayerCamera("Mid", "RenderMid", false);
        farSlot = CreateLayerCamera("Far", "RenderFar", true);
    }

    private LayerSlot CreateLayerCamera(string displayName, string unityLayerName, bool rendersSky)
    {
        int layer = LayerMask.NameToLayer(unityLayerName);
        if (layer < 0)
        {
            Debug.LogError($"[SenderFramePacker] Required layer '{unityLayerName}' was not found.");
            return null;
        }

        var cameraObject = new GameObject($"{displayName}LayerCamera_Runtime")
        {
            hideFlags = HideFlags.DontSave
        };
        cameraObject.transform.SetParent(transform, false);

        Camera layerCamera = cameraObject.AddComponent<Camera>();
        if (renderCamera != null)
            layerCamera.CopyFrom(renderCamera);
        UniversalAdditionalCameraData layerCameraData =
            layerCamera.GetUniversalAdditionalCameraData();
        layerCameraData.renderType = CameraRenderType.Base;
        layerCameraData.requiresDepthTexture = true;
        if (renderCamera != null &&
            renderCamera.TryGetComponent(out UniversalAdditionalCameraData sourceCameraData))
        {
            layerCameraData.renderPostProcessing = sourceCameraData.renderPostProcessing;
            layerCameraData.antialiasing = sourceCameraData.antialiasing;
            layerCameraData.antialiasingQuality = sourceCameraData.antialiasingQuality;
            layerCameraData.stopNaN = sourceCameraData.stopNaN;
            layerCameraData.dithering = sourceCameraData.dithering;
        }
        layerCamera.enabled = false;
        layerCamera.cullingMask = 1 << layer;
        layerCamera.depthTextureMode |= DepthTextureMode.Depth;
        layerCamera.clearFlags = rendersSky ? CameraClearFlags.Skybox : CameraClearFlags.SolidColor;
        layerCamera.backgroundColor = new Color(0, 0, 0, 0);

        LayerFrameCapture capture = cameraObject.AddComponent<LayerFrameCapture>();
        capture.packMaterial = packMaterial;

        return new LayerSlot
        {
            name = displayName,
            layer = layer,
            rendersSky = rendersSky,
            cameraObject = cameraObject,
            camera = layerCamera,
            capture = capture,
        };
    }

    public void ConfigureLayers(
        bool nearActive, Vector2Int nearResolution,
        bool midActive, Vector2Int midResolution,
        bool farActive, Vector2Int farResolution)
    {
        ConfigureLayer(nearSlot, nearActive, nearResolution);
        ConfigureLayer(midSlot, midActive, midResolution);
        ConfigureLayer(farSlot, farActive, farResolution);

        int width = 0;
        int height = 0;
        AccumulateCanvas(nearSlot, ref width, ref height);
        AccumulateCanvas(midSlot, ref width, ref height);
        AccumulateCanvas(farSlot, ref width, ref height);
        CurrentColorResolution = new Vector2Int(width, height);
        streamWidth = width;
        streamHeight = height;
    }

    // Backward-compatible entry point used before the first scheduler packet.
    public void SetRenderResolution(Vector2Int colorResolution)
    {
        ConfigureLayers(
            IncludesLayer(initialCullingMask, nearSlot), colorResolution,
            IncludesLayer(initialCullingMask, midSlot), colorResolution,
            IncludesLayer(initialCullingMask, farSlot), colorResolution);
    }

    private void ConfigureLayer(LayerSlot slot, bool active, Vector2Int requestedResolution)
    {
        if (slot == null)
            return;

        Vector2Int resolution = ClampResolution(requestedResolution);
        if (active && (slot.colorRT == null || slot.resolution != resolution))
            AllocateLayerTextures(slot, resolution);

        slot.active = active;
        slot.camera.enabled = active;
        slot.capture.packedRT = slot.packedLayerRT;
        slot.capture.packMaterial = packMaterial;
    }

    private void AllocateLayerTextures(LayerSlot slot, Vector2Int resolution)
    {
        ReleaseTexture(ref slot.colorRT);
        ReleaseTexture(ref slot.packedLayerRT);

        slot.colorRT = new RenderTexture(
            resolution.x, resolution.y, 24, RenderTextureFormat.ARGB32)
        {
            name = $"{slot.name}Color_{resolution.x}x{resolution.y}",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
            autoGenerateMips = false
        };
        slot.colorRT.Create();

        slot.packedLayerRT = new RenderTexture(
            resolution.x, resolution.y * 2, 0, RenderTextureFormat.ARGB32)
        {
            name = $"{slot.name}Packed_{resolution.x}x{resolution.y * 2}",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
            autoGenerateMips = false
        };
        slot.packedLayerRT.Create();

        slot.camera.targetTexture = slot.colorRT;
        slot.capture.packedRT = slot.packedLayerRT;
        slot.resolution = resolution;
        Debug.Log($"[SenderFramePacker] {slot.name} layer rendering at {resolution.x}x{resolution.y}.");
    }

    private void SyncLayerCamera(LayerSlot slot, int order)
    {
        if (slot == null || renderCamera == null || slot.camera == null)
            return;

        bool enabled = slot.camera.enabled;
        RenderTexture target = slot.colorRT;
        slot.camera.CopyFrom(renderCamera);
        slot.camera.enabled = enabled;
        slot.camera.targetTexture = target;
        slot.camera.cullingMask = 1 << slot.layer;
        slot.camera.depth = renderCamera.depth + order;
        slot.camera.depthTextureMode |= DepthTextureMode.Depth;
        slot.camera.clearFlags = slot.rendersSky ? CameraClearFlags.Skybox : CameraClearFlags.SolidColor;
        slot.camera.backgroundColor = new Color(0, 0, 0, 0);
    }

    private void CompositeSharedPackedTexture()
    {
        if (packedRT == null)
            return;

        // Keep a visible diagnostic/fallback path even if a player was built
        // without the composition shader. A missing shader must not silently
        // turn every local/remote stream black.
        if (layerCompositeMaterial == null)
        {
            Texture fallback = FirstActiveLayerTexture();
            Graphics.Blit(fallback ?? Texture2D.blackTexture, packedRT);
            return;
        }

        layerCompositeMaterial.SetTexture("_NearTex", TextureFor(nearSlot));
        layerCompositeMaterial.SetTexture("_MidTex", TextureFor(midSlot));
        layerCompositeMaterial.SetTexture("_FarTex", TextureFor(farSlot));
        layerCompositeMaterial.SetFloat("_NearActive", IsActive(nearSlot) ? 1f : 0f);
        layerCompositeMaterial.SetFloat("_MidActive", IsActive(midSlot) ? 1f : 0f);
        layerCompositeMaterial.SetFloat("_FarActive", IsActive(farSlot) ? 1f : 0f);

        Graphics.Blit(Texture2D.blackTexture, packedRT, layerCompositeMaterial, 0);
    }

    private Texture FirstActiveLayerTexture()
    {
        if (IsActive(nearSlot)) return nearSlot.packedLayerRT;
        if (IsActive(midSlot)) return midSlot.packedLayerRT;
        if (IsActive(farSlot)) return farSlot.packedLayerRT;
        return null;
    }

    public bool OwnsLayerCamera(Camera camera)
    {
        return camera != null &&
            ((nearSlot != null && camera == nearSlot.camera) ||
             (midSlot != null && camera == midSlot.camera) ||
             (farSlot != null && camera == farSlot.camera));
    }

    private Vector2Int ClampResolution(Vector2Int resolution)
    {
        return new Vector2Int(
            Mathf.Clamp(resolution.x, 2, maximumColorResolution.x),
            Mathf.Clamp(resolution.y, 2, maximumColorResolution.y));
    }

    private static bool IncludesLayer(int mask, LayerSlot slot)
    {
        return slot != null && (mask & (1 << slot.layer)) != 0;
    }

    private static bool IsActive(LayerSlot slot)
    {
        return slot != null && slot.active && slot.packedLayerRT != null;
    }

    private static Texture TextureFor(LayerSlot slot)
    {
        return IsActive(slot) ? slot.packedLayerRT : Texture2D.blackTexture;
    }

    private static void AccumulateCanvas(LayerSlot slot, ref int width, ref int height)
    {
        if (!IsActive(slot))
            return;
        width = Mathf.Max(width, slot.resolution.x);
        height = Mathf.Max(height, slot.resolution.y);
    }

    private static void ReleaseTexture(ref RenderTexture texture)
    {
        if (texture == null)
            return;
        texture.Release();
        Destroy(texture);
        texture = null;
    }

    private void ReleaseSlot(LayerSlot slot)
    {
        if (slot == null)
            return;
        if (slot.camera != null)
            slot.camera.targetTexture = null;
        ReleaseTexture(ref slot.colorRT);
        ReleaseTexture(ref slot.packedLayerRT);
        if (slot.cameraObject != null)
            Destroy(slot.cameraObject);
    }

    private void OnDestroy()
    {
        ReleaseSlot(nearSlot);
        ReleaseSlot(midSlot);
        ReleaseSlot(farSlot);

        if (runtimePackedRT != null)
        {
            runtimePackedRT.Release();
            Destroy(runtimePackedRT);
            runtimePackedRT = null;
        }

        if (packMaterial != null)
            Destroy(packMaterial);
        if (layerCompositeMaterial != null)
            Destroy(layerCompositeMaterial);
    }
}
