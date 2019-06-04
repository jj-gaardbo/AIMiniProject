using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Globals{
    public static string moveDir;
    public static GameObject player;
    public static Transform playerInitialTransform;
    public static Vector3 playerInitialPosition;
    public static Quaternion playerInitialRotation;
    public static int numWalls = 0;

    public static float reward = 1.0f;
    public static bool hasReachedGoal = false;
    public static float[] rays;
    public static bool done;
    public static float distanceToGoal;
    public static float originalDistanceToGoal;
    public static int goalCount;

    public static void setGoalCount(int count){
        goalCount = count;
    }

    public static void addANewWall(){
        numWalls++;
    }

    public static void updateReward(float rew){
        reward = rew;
    }

    public static void setDistanceToGoal(float distance){
        distanceToGoal = distance;
    }

    public static void setOriginalDistanceToGoal(float distance){
        originalDistanceToGoal = distance;
    }

    public static float[] getRays(){
        return rays;
    }

    public static void setRays(float[] raysArray){
        rays = raysArray;
    }

    public static void goalReached(){
        hasReachedGoal = true;
    }

    public static void setMoveDir(string dir){
        moveDir = dir;
    }

    public static void setPlayer(GameObject playerObject, Transform initialTransform, Vector3 initialPosition, Quaternion initialRotation){
        player = playerObject;
        playerInitialTransform = initialTransform;
        playerInitialPosition = initialPosition;
        playerInitialRotation = initialRotation;
    }

    public static GameObject getPlayer(){
        return player;
    }

    public static void resetPlayer(){
        player.transform.position = playerInitialPosition;
        player.transform.rotation = playerInitialRotation;
    }


}
