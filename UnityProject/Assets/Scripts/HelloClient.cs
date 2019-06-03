using UnityEngine;
using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using System.Collections;
using System.IO;
using System.Text;

public class HelloClient : MonoBehaviour
{
    private HelloRequester _helloRequester;
    public bool SendPack = true;
    private byte[] current_img;

    private void OnPostRender()
    {

        if (SendPack)
        {
            Message send = new Message(Globals.moveDir, Globals.reward, Globals.hasReachedGoal, Globals.rays, Globals.done, Globals.distanceToGoal, Globals.originalDistanceToGoal, false, Globals.goalCount);

            byte[] array = Encoding.ASCII.GetBytes(JsonUtility.ToJson(send));
            _helloRequester.bytes = array;
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