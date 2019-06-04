episodes = monitor[,1]
time = monitor[,2]
add_rew = monitor[,3]
mean_rew = monitor[,4]


plot(time, main="Time alive",
     ylab="Time", xlab="Episode")
lines(time)

plot(add_rew, main="Reward (Added)",
     ylab="Reward", xlab="Episode")
lines(add_rew)

plot(mean_rew, main="Reward (Mean)",
     ylab="Reward", xlab="Episode")
lines(mean_rew)