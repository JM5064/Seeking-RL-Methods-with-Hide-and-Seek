using Unity.MLAgents.Sensors;
using UnityEngine;

public class CameraTextureSensorComponent : SensorComponent
{
    [Header("Camera Sensor Settings")]
    public Camera agentCamera;
    public RenderTexture renderTexture;
    public string sensorName = "CameraTextureSensor";
    public int width = 32;
    public int height = 32;

    private CameraTextureSensor sensor;

    public override ISensor[] CreateSensors()
    {
        if (agentCamera == null)
            Debug.LogError("CameraTextureSensorComponent: No camera assigned!");
        if (renderTexture == null)
            Debug.LogError("CameraTextureSensorComponent: No RenderTexture assigned!");

        sensor = new CameraTextureSensor(agentCamera, renderTexture, sensorName, width, height);
        return new ISensor[] { sensor };
    }
}