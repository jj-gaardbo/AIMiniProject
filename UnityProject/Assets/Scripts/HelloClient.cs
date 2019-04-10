using UnityEngine;
using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using System.Collections;
using System.IO;

public class HelloClient : MonoBehaviour
{
    private HelloRequester _helloRequester;
    public bool SendPack = true;
    private byte[] current_img;

    private void OnPostRender()
    {

        if (SendPack)
        {
            //Create a new texture with the width and height of the screen
            Texture2D texture = new Texture2D(256, 256, TextureFormat.RGB24, false);
            //Read the pixels in the Rect starting at 0,0 and ending at the screen's width and height
            texture.ReadPixels(new Rect(0, 0, 256, 256), 0, 0, false);
            texture.Apply();
            //Byte encode the image
            current_img = texture.EncodeToPNG();
            //Destroy the tex for mem leak control
            Object.Destroy(texture);

            // For testing purposes, also write to a file in the project folder, if you want to save the image directly from Unity, use this.
            //File.WriteAllBytes(Application.dataPath + "SavedScreen.png", current_img);

            _helloRequester.bytes = current_img;
            _helloRequester.Continue();
        }
        else if (!SendPack)
        {
            _helloRequester.Pause();

        }
    }

    private void Start()
    {
        _helloRequester = new HelloRequester();
        _helloRequester.Start();
    }

    private void OnDestroy()
    {
        _helloRequester.Stop();
    }


}