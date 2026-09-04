using UnityEngine;

/// <summary>
/// Marks a runtime layer camera whose color and depth must be packed by
/// PackDepthRendererFeature. SenderFramePacker owns the referenced resources.
/// </summary>
public sealed class LayerFrameCapture : MonoBehaviour
{
    [HideInInspector] public RenderTexture packedRT;
    [HideInInspector] public Material packMaterial;
}
