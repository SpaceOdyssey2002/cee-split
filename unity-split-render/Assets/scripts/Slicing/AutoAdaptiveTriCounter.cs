using System.Collections.Generic;
using System.Globalization;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class AutoAdaptiveTriCounter : MonoBehaviour
{
    [Header("Core Settings")]
    public Camera mainCam;
    [Range(0.1f, 0.5f)] public float targetRatioNear = 0.33f;
    [Range(0.1f, 0.5f)] public float targetRatioMid = 0.33f;

    [Header("Runtime Results")]
    public int trisNear;
    public int trisMid;
    public int trisFar;
    public float currentSplit_Near;
    public float currentSplit_Mid;

    [Header("Network and Node")]
    public RenderStreamingMonitor localMonitor;
    public int myNodeID = 0;
    public string targetIP = "100.113.34.83";
    public int targetPort = 9998;

    [Header("UI Settings")]
    public bool showUI = true;

    private class RenderData
    {
        public Renderer renderer;
        public int triCount;
        public float currentDist;
        public int lastLayer = -1;
    }

    private readonly List<RenderData> allObjects = new List<RenderData>();
    private Plane[] frustumPlanes;
    private float updateTimer;
    private const float UpdateInterval = 0.2f;
    private int layerNear;
    private int layerMid;
    private int layerFar;
    private string uiNear;
    private string uiMid;
    private string uiFar;
    private string uiTotal;
    private UdpClient udpSender;

    public int TotalTris => trisNear + trisMid + trisFar;

    private void Start()
    {
        if (mainCam == null)
            mainCam = Camera.main;

        layerNear = LayerMask.NameToLayer("RenderNear");
        layerMid = LayerMask.NameToLayer("RenderMid");
        layerFar = LayerMask.NameToLayer("RenderFar");
        if (layerNear < 0 || layerMid < 0 || layerFar < 0)
        {
            Debug.LogError("[TriCounter] RenderNear, RenderMid and RenderFar layers are required.");
            enabled = false;
            return;
        }

        udpSender = new UdpClient { EnableBroadcast = true };
        RefreshSceneObjects();
    }

    public void RefreshSceneObjects()
    {
        allObjects.Clear();
        MeshFilter[] meshes = Object.FindObjectsByType<MeshFilter>(FindObjectsSortMode.None);

        foreach (MeshFilter meshFilter in meshes)
        {
            if (meshFilter.sharedMesh == null)
                continue;

            Renderer renderer = meshFilter.GetComponent<Renderer>();
            if (renderer == null || !renderer.enabled)
                continue;

            allObjects.Add(new RenderData
            {
                renderer = renderer,
                triCount = meshFilter.sharedMesh.triangles.Length / 3
            });
        }
    }

    private void Update()
    {
        updateTimer += Time.deltaTime;
        if (updateTimer < UpdateInterval)
            return;

        updateTimer = 0f;
        CalculateMetrics();
    }

    private void CalculateMetrics()
    {
        if (mainCam == null)
            return;

        frustumPlanes = GeometryUtility.CalculateFrustumPlanes(mainCam);
        Vector3 camPos = mainCam.transform.position;
        var visible = new List<RenderData>();
        long totalTris = 0;
        float maxDist = 0f;

        foreach (RenderData obj in allObjects)
        {
            if (obj.renderer == null ||
                !GeometryUtility.TestPlanesAABB(frustumPlanes, obj.renderer.bounds))
                continue;

            Vector3 closest = obj.renderer.bounds.ClosestPoint(camPos);
            obj.currentDist = Vector3.Distance(camPos, closest);
            maxDist = Mathf.Max(maxDist, obj.currentDist);
            visible.Add(obj);
            totalTris += obj.triCount;
        }

        if (totalTris == 0)
            return;

        visible.Sort((a, b) => a.currentDist.CompareTo(b.currentDist));

        long sum = 0;
        float rawNear = maxDist;
        float rawMid = maxDist;
        foreach (RenderData obj in visible)
        {
            sum += obj.triCount;
            if (sum >= totalTris * targetRatioNear && Mathf.Approximately(rawNear, maxDist))
                rawNear = obj.currentDist;
            if (sum >= totalTris * (targetRatioNear + targetRatioMid))
            {
                rawMid = obj.currentDist;
                break;
            }
        }

        currentSplit_Near = Mathf.Lerp(currentSplit_Near, rawNear, 0.5f);
        currentSplit_Mid = Mathf.Lerp(currentSplit_Mid, Mathf.Max(rawMid, currentSplit_Near), 0.5f);

        int nearTris = 0;
        int midTris = 0;
        int farTris = 0;
        foreach (RenderData obj in visible)
        {
            int targetLayer = obj.currentDist < currentSplit_Near
                ? layerNear
                : obj.currentDist < currentSplit_Mid ? layerMid : layerFar;

            if (obj.lastLayer != targetLayer)
            {
                obj.renderer.gameObject.layer = targetLayer;
                obj.lastLayer = targetLayer;
            }

            if (targetLayer == layerNear)
                nearTris += obj.triCount;
            else if (targetLayer == layerMid)
                midTris += obj.triCount;
            else
                farTris += obj.triCount;
        }

        trisNear = nearTris;
        trisMid = midTris;
        trisFar = farTris;

        uiNear = $"Near: {nearTris / 1000f:F1}k | {currentSplit_Near:F1}m";
        uiMid = $"Mid:  {midTris / 1000f:F1}k | {currentSplit_Mid:F1}m";
        uiFar = $"Far:  {farTris / 1000f:F1}k";
        uiTotal = $"Total: {(nearTris + midTris + farTris) / 1000f:F1}k";

        SendStatsViaUDP(nearTris, midTris, farTris);
    }

    private void SendStatsViaUDP(int nearTris, int midTris, int farTris)
    {
        if (udpSender == null)
            return;

        try
        {
            double encodeTime = localMonitor != null ? localMonitor.GetEncodeTime() : 0.0;
            double frameTime = localMonitor != null ? localMonitor.GetFrameTimeMs() : 0.0;
            // The monitor is initialized asynchronously with the WebRTC connection.
            // Keep the render-frame sample useful during that short startup window.
            if (!(frameTime > 0.0) || double.IsNaN(frameTime) || double.IsInfinity(frameTime))
                frameTime = Time.unscaledDeltaTime > 0.0f
                    ? Time.unscaledDeltaTime * 1000.0
                    : 0.0;
            double bandwidth = localMonitor != null
                ? localMonitor.GetAvailableOutgoingBandwidth()
                : 0.0;
            int bandwidthValid = localMonitor != null && localMonitor.HasAvailableOutgoingBandwidth()
                ? 1
                : 0;
            string json = "{" +
                "\"n\":" + nearTris + "," +
                "\"m\":" + midTris + "," +
                "\"f\":" + farTris + "," +
                "\"node\":" + myNodeID + "," +
                "\"enc\":" + encodeTime.ToString("F1", CultureInfo.InvariantCulture) + "," +
                "\"rft\":" + frameTime.ToString("F1", CultureInfo.InvariantCulture) + "," +
                "\"bw\":" + bandwidth.ToString("F3", CultureInfo.InvariantCulture) + "," +
                "\"bw_valid\":" + bandwidthValid +
                "}";

            byte[] data = Encoding.UTF8.GetBytes(json);
            udpSender.Send(data, data.Length, targetIP, targetPort);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[TriCounter] UDP send failed: " + e.Message);
        }
    }

    private void OnGUI()
    {
        if (!showUI)
            return;

        GUI.Box(new Rect(10, 10, 220, 100), string.Empty);
        GUI.Label(new Rect(20, 15, 200, 20), uiNear);
        GUI.Label(new Rect(20, 35, 200, 20), uiMid);
        GUI.Label(new Rect(20, 55, 200, 20), uiFar);
        GUI.Label(new Rect(20, 75, 200, 20), uiTotal);
    }

    private void OnDestroy()
    {
        udpSender?.Close();
    }
}
