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
    public bool addWall;
    public int goalCount;

    public Message(string m, float r, bool g, float[] rayArray, bool d, float distance, float origDistance, bool addNewWall, int count){
        move = m;
        reward = r;
        hasReachedGoal = g;
        rays = rayArray;
        done = d;
        distanceToGoal = distance;
        originalDistanceToGoal = origDistance;
        addWall = addNewWall;
        goalCount = count;
    }
}

