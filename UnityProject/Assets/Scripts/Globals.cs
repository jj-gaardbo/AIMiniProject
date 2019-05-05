using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Globals{
    public static string moveDir;
    public static GameObject player;
    public static float reward = 1.0f;
    public static bool hasReachedGoal = false;
    public static float[] rays;
    public static bool done;
    public static float distanceToGoal;
    public static float originalDistanceToGoal;

    public static void updateReward(float rew){
        reward = rew;
    }

    public static void setDistanceToGoal(float distance){
        distanceToGoal = distance;
    }

    public static void setOriginalDistanceToGoal(float distance){
        originalDistanceToGoal = distance;
    }

    public static void setDone(){
        done = true;
    }

    public static float[] getRays(){
        return rays;
    }

    public static void setRays(float[] raysArray){
        rays = raysArray;
    }

    public static void goalReached(){
        hasReachedGoal = true;
        setDone();
        Debug.Log("Goal reached " + hasReachedGoal);
    }

    public static void setMoveDir(string dir){
        moveDir = dir;
        Debug.Log("MoveDir " + moveDir);
    }

    public static void setPlayer(GameObject playerObject){
        player = playerObject;
    }

    public static GameObject getPlayer(){
        return player;
    }
}
