using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.RenderGraphModule.Util;

public class PackDepthRendererFeature : ScriptableRendererFeature
{
    private readonly Dictionary<int, PackDepthRenderPass> passes =
        new Dictionary<int, PackDepthRenderPass>();

    public override void Create()
    {
        foreach (PackDepthRenderPass existingPass in passes.Values)
            existingPass.Cleanup();
        passes.Clear();
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (renderingData.cameraData.cameraType != CameraType.Game) return;

        Camera camera = renderingData.cameraData.camera;
        if (!camera.enabled || camera.cullingMask == 0) return;

        int cameraId = camera.GetInstanceID();

        RenderTexture packedTarget = null;
        Material packingMaterial = null;

        var packer = camera.GetComponent<SenderFramePacker>();
        if (packer != null)
        {
            packedTarget = packer.packedRT;
            packingMaterial = packer.packMaterial;
        }
        else
        {
            var layerCapture = camera.GetComponent<LayerFrameCapture>();
            if (layerCapture != null)
            {
                packedTarget = layerCapture.packedRT;
                packingMaterial = layerCapture.packMaterial;
            }
        }

        if (packedTarget != null && packingMaterial != null)
        {
            if (!passes.TryGetValue(cameraId, out PackDepthRenderPass pass))
            {
                pass = new PackDepthRenderPass
                {
                    renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing
                };
                passes.Add(cameraId, pass);
            }

            pass.Setup(packedTarget, packingMaterial);
            renderer.EnqueuePass(pass);
        }
    }

    protected override void Dispose(bool disposing)
    {
        foreach (PackDepthRenderPass pass in passes.Values)
            pass.Cleanup();
        passes.Clear();
    }
}

public class PackDepthRenderPass : ScriptableRenderPass
{
    private static readonly int CameraDepthTextureId =
        Shader.PropertyToID("_CameraDepthTexture");

    private RenderTexture targetRT;
    private Material material;
    private RTHandle targetRTHandle;

    public void Setup(RenderTexture rt, Material mat)
    {
        targetRT = rt;
        material = mat;

        // 声明需要颜色和深度
        ConfigureInput(ScriptableRenderPassInput.Color | ScriptableRenderPassInput.Depth);

        // 🌟【关键修复1】只有当 RT 改变时，才重新分配 Handle！绝不能每帧 Alloc！
        if (targetRT != null)
        {
            if (targetRTHandle == null || targetRTHandle.rt != targetRT)
            {
                if (targetRTHandle != null) RTHandles.Release(targetRTHandle);
                targetRTHandle = RTHandles.Alloc(targetRT);
            }
        }
    }

    private class PassData
    {
        public Material material;
        public TextureHandle sourceColor;
        public TextureHandle sourceDepth;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        if (material == null || targetRTHandle == null) return;

        var resourceData = frameData.Get<UniversalResourceData>();

        // 🌟【防黑屏警告】如果没拿到深度图，在控制台报错，而不是默默黑屏
        if (!resourceData.cameraColor.IsValid() || !resourceData.cameraDepth.IsValid())
        {
            Debug.LogWarning("[PackDepth] 没有获取到有效的颜色或深度贴图！请检查 URP 设置是否开启了 Depth。");
            return;
        }

        TextureHandle sourceColor = resourceData.cameraColor;
        TextureHandle sourceDepth = resourceData.cameraDepth;
        TextureHandle destination = renderGraph.ImportTexture(targetRTHandle);

        using (var builder = renderGraph.AddRasterRenderPass<PassData>("PackRgbDepth", out var passData))
        {
            passData.material = material;
            passData.sourceColor = sourceColor;
            passData.sourceDepth = sourceDepth;

            builder.UseTexture(passData.sourceColor, AccessFlags.Read);
            builder.UseTexture(passData.sourceDepth, AccessFlags.Read);

            builder.SetRenderAttachment(destination, 0);
            // The destination is an imported texture consumed by WebRTC or the
            // terminal compositor outside RenderGraph, so the graph cannot infer
            // the dependency on its own.
            builder.AllowPassCulling(false);
            builder.AllowGlobalStateModification(true);

            builder.SetRenderFunc((PassData data, RasterGraphContext context) =>
            {
                // Bind the current camera's RenderGraph depth explicitly. Setting
                // a transient TextureHandle directly on Material can leave the
                // shader sampling an invalid/previous camera texture.
                context.cmd.SetGlobalTexture(CameraDepthTextureId, data.sourceDepth);
                Blitter.BlitTexture(context.cmd, data.sourceColor, new Vector4(1, 1, 0, 0), data.material, 0);
            });
        }
    }

    public void Cleanup()
    {
        if (targetRTHandle != null)
        {
            targetRTHandle.Release();
            targetRTHandle = null;
        }
    }
}
