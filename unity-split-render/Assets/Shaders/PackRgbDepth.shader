Shader "Custom/PackRgbDepth"
{
    Properties
    {
        // URP 的 Blitter 会自动处理输入贴图，这里不需要暴露 _MainTex 了
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline" = "UniversalPipeline"}
        ZTest Always Cull Off ZWrite Off

        Pass
        {
            Name "PackRgbDepthPass"

            HLSLPROGRAM
            // 使用 URP 内置的顶点着色器 Vert，我们只需要写片段着色器 Frag 即可
            #pragma vertex Vert
            #pragma fragment Frag
            
            // 引入 URP 核心库和全屏 Blit 库
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            // 声明接收深度的纹理 (C# 脚本里通过 SetTexture 绑定的就是这个)
            TEXTURE2D(_CameraDepthTexture);
            SAMPLER(sampler_CameraDepthTexture);

            // 注意：_BlitTexture 是 Blit.hlsl 内部已经声明好的，我们直接用就行

            half4 Frag(Varyings input) : SV_Target
            {
                // 获取当前像素的 UV
                float2 uv = input.texcoord;
                
                if (uv.y > 0.5) 
                {
                    // 上半部分 (y > 0.5)：绘制彩色 RGB
                    // 将 y 从 [0.5, 1.0] 映射回[0.0, 1.0]
                    float2 rgbUV = float2(uv.x, (uv.y - 0.5) * 2.0);
                    
                    // 采样颜色 (使用 URP 内置的 sampler_LinearClamp 采样器)
                    return SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearClamp, rgbUV);
                }
                else 
                {
                    // 下半部分 (y < 0.5)：绘制深度图
                    // 将 y 从[0.0, 0.5] 映射回[0.0, 1.0]
                    float2 depthUV = float2(uv.x, uv.y * 2.0);
                    
                    // 采样深度值
                    float rawDepth = SAMPLE_TEXTURE2D(_CameraDepthTexture, sampler_CameraDepthTexture, depthUV).r;
                    
                    // 使用完整相机裁剪范围编码对数眼空间深度。相比 Linear01，
                    // 对数编码在 8-bit 视频流里能保留更多近中景深度精度。
                    float eyeDepth = LinearEyeDepth(rawDepth, _ZBufferParams);
                    float nearPlane = max(_ProjectionParams.y, 1e-4);
                    float farPlane = max(_ProjectionParams.z, nearPlane + 1e-3);
                    float logRange = log2(farPlane / nearPlane);
                    float encodedDepth = saturate(log2(max(eyeDepth, nearPlane) / nearPlane) / logRange);

                    return half4(encodedDepth, encodedDepth, encodedDepth, 1.0);
                }
            }
            ENDHLSL
        }
    }
}
