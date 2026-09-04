Shader "Custom/DepthComposite_Bad"
{
    Properties
    {
        _FarTex  ("Far Layer",  2D) = "black" {}
        _MidTex  ("Mid Layer",  2D) = "black" {}
        _NearTex ("Near Layer", 2D) = "black" {}
        // 黑色背景的硬抠图阈值
        _Threshold ("Black Threshold", Range(0, 1)) = 0.1
    }
    SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _FarTex;
            sampler2D _MidTex;
            sampler2D _NearTex;
            float _Threshold;

            // 辅助函数：只提取视频流上半部分的 RGB 画面（故意丢弃下半部的深度图）
            fixed3 GetRGB(sampler2D tex, float2 uv)
            {
                float2 uv_rgb = float2(uv.x, uv.y * 0.5 + 0.5);
                return tex2D(tex, uv_rgb).rgb;
            }

            fixed4 frag(v2f_img i) : SV_Target
            {
                fixed3 farRGB  = GetRGB(_FarTex, i.uv);
                fixed3 midRGB  = GetRGB(_MidTex, i.uv);
                fixed3 nearRGB = GetRGB(_NearTex, i.uv);

                fixed3 result = farRGB;

                // 【硬抠图逻辑】：计算颜色向量的长度，如果接近 0（黑色），就是背景
                // step(a, b) 函数：如果 b < a 返回 0，否则返回 1。这会导致极其严重的狗牙锯齿！
                
                float midVisible = step(_Threshold, length(midRGB));
                result = lerp(result, midRGB, midVisible);

                float nearVisible = step(_Threshold, length(nearRGB));
                result = lerp(result, nearRGB, nearVisible);

                return fixed4(result, 1.0);
            }
            ENDCG
        }
    }
}