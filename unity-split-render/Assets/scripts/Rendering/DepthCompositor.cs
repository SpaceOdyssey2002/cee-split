using Unity.RenderStreaming;
using UnityEngine;
using UnityEngine.UI;

public class DepthCompositor : MonoBehaviour
{
    [Header("Remote streams (Node0 / Node1)")]
    [SerializeField] private VideoStreamReceiver receiverNode0;
    [SerializeField] private VideoStreamReceiver receiverNode1;

    [Header("Legacy/local stream fallback (Node2)")]
    [Tooltip("Kept for existing receiver scenes. A local SenderFramePacker takes priority when present.")]
    [SerializeField] private VideoStreamReceiver receiverNode2;

    [Header("Local render (Node2)")]
    [SerializeField] private RenderTexture localNode2Texture;
    [SerializeField] private SenderFramePacker localNode2Packer;

    [Header("Camera")]
    [Tooltip("All three rendering nodes must use the same near/far clip planes.")]
    [SerializeField] private Camera mainCam;

    [Header("Output")]
    [SerializeField] private RawImage outputImage;
    [SerializeField] private Shader compositeShader;
    [SerializeField] private Vector2Int singleLayerResolution = new Vector2Int(1920, 1080);

    [Header("Depth composition")]
    [SerializeField, Range(0.9f, 1f)] private float backgroundDepthThreshold = 0.995f;

    [Header("Performance")]
    [SerializeField, Min(1)] private int targetCompositeFPS = 60;

    private float compositeTimer;
    private RenderTexture compositeRT;
    private Material compositeMat;
    private int nearNodeIdx = -1;
    private int midNodeIdx = -1;
    private int farNodeIdx = -1;

    private void Start()
    {
        compositeRT = new RenderTexture(
            singleLayerResolution.x,
            singleLayerResolution.y,
            0,
            RenderTextureFormat.ARGB32)
        {
            name = "DepthCompositeOutput"
        };
        compositeRT.Create();

        if (outputImage != null)
            outputImage.texture = compositeRT;

        if (compositeShader == null)
            compositeShader = Shader.Find("Custom/DepthCompositePacked");

        if (compositeShader != null)
            compositeMat = new Material(compositeShader);
        else
            Debug.LogError("[DepthCompositor] Custom/DepthCompositePacked shader was not found.");

        if (mainCam == null)
            mainCam = Camera.main;

        if (localNode2Packer == null && mainCam != null)
            localNode2Packer = mainCam.GetComponent<SenderFramePacker>();
    }

    private void LateUpdate()
    {
        if (nearNodeIdx == -1)
            return;

        compositeTimer += Time.deltaTime;
        float compositeInterval = 1f / Mathf.Max(1, targetCompositeFPS);
        if (compositeTimer < compositeInterval)
            return;

        compositeTimer %= compositeInterval;

        Texture nearTex = GetTextureByNode(nearNodeIdx);
        Texture midTex = GetTextureByNode(midNodeIdx);
        Texture farTex = GetTextureByNode(farNodeIdx);

        // A build-time shader stripping mistake should still show the received
        // frame for diagnosis instead of leaving the output permanently black.
        if (compositeMat == null)
        {
            Texture fallback = nearTex ?? midTex ?? farTex;
            Graphics.Blit(fallback ?? Texture2D.blackTexture, compositeRT);
            return;
        }

        compositeMat.SetTexture(
            "_NearTex",
            nearTex != null ? nearTex : Texture2D.blackTexture);
        compositeMat.SetTexture(
            "_MidTex",
            midTex != null ? midTex : Texture2D.blackTexture);
        compositeMat.SetTexture(
            "_FarTex",
            farTex != null ? farTex : Texture2D.blackTexture);

        compositeMat.SetFloat("_HasNear", nearTex != null ? 1f : 0f);
        compositeMat.SetFloat("_HasMid", midTex != null ? 1f : 0f);
        compositeMat.SetFloat("_HasFar", farTex != null ? 1f : 0f);

        UpdateDepthSettings();

        Texture sourceTex = nearTex ?? midTex ?? farTex ?? Texture2D.blackTexture;
        Graphics.Blit(sourceTex, compositeRT, compositeMat);
    }

    private void UpdateDepthSettings()
    {
        if (mainCam == null)
            return;

        compositeMat.SetVector(
            "_CameraClip",
            new Vector4(mainCam.nearClipPlane, mainCam.farClipPlane, 0f, 0f));
        compositeMat.SetFloat("_BackgroundDepthThreshold", backgroundDepthThreshold);
    }

    public void UpdateLayerMapping(int nearNode, int midNode, int farNode)
    {
        nearNodeIdx = nearNode;
        midNodeIdx = midNode;
        farNodeIdx = farNode;
    }

    private Texture GetTextureByNode(int nodeIdx)
    {
        if (nodeIdx == 0)
            return receiverNode0 != null ? receiverNode0.texture : null;

        if (nodeIdx == 1)
            return receiverNode1 != null ? receiverNode1.texture : null;

        if (nodeIdx == 2)
        {
            if (localNode2Packer != null && localNode2Packer.packedRT != null)
                return localNode2Packer.packedRT;
            if (localNode2Texture != null)
                return localNode2Texture;
            return receiverNode2 != null ? receiverNode2.texture : null;
        }

        return null;
    }

    public void UpdateLocalTexture(RenderTexture newLocalRT)
    {
        localNode2Texture = newLocalRT;
    }

    private void OnDestroy()
    {
        if (compositeRT != null)
        {
            compositeRT.Release();
            Destroy(compositeRT);
        }

        if (compositeMat != null)
            Destroy(compositeMat);
    }
}
