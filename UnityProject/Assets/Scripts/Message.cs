using System;
using UnityEngine;

[Serializable]
public class Message{
    public string move;
    public float reward;
    public bool hasReachedGoal;
    public float[] rays;
    public bool done;
    public float distanceToGoal;
    public float originalDistanceToGoal;

    public Message(string m, float r, bool g, float[] rayArray, bool d, float distance, float origDistance){
        move = m;
        reward = r;
        hasReachedGoal = g;
        rays = rayArray;
        done = d;
        distanceToGoal = distance;
        originalDistanceToGoal = origDistance;
    }
}

