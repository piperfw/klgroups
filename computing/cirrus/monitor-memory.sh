#! /usr/bin/env bash
# usage: 
#./monitor-memory.sh PID
# where PID is target process
# e.g. monitor memory of most recent srun command:  
# srun --cpu-bind=cores run-me.py &
# ./monitor=memory.sh $!
echo $0 "for" $(whoami)"@""$HOSTNAME"
for i in {1..3000}
do
		a=$(date +"%Y-%m-%d %H:%M:%S")
		# rss is resident set size for current shell process
		b=$(ps -o rss= $1 | awk '{printf "%.1f\n", $1 / 1024}') # convert to MB
		if [[ -z "$b" ]]; then 
				# if $1 not an existing process, ps returns empty string
				echo "process $1 not found, exiting $0"
				break
		else
				# otherwise print timestamp and memory usage of $1
				echo "$a usage $b MB"
		fi
		# run every minute
		sleep 60
done
