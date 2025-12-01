#!/bin/bash

JOBS_WHICH_HAVE_STARTED_RUNNING=()
JOBS_WHICH_HAVE_FINISHED=()
CLUSTER_NAME="KILLARNEY"

# function which runs $(sq | awk '$5 == "R" && $4 ~ /multi_runner/ {print $1}')
currently_running_job() {
  echo $(sq | awk '$5 == "R" && $4 ~ /sequential_job/ {print $1}')
}

request_job() {
  echo "requesting job ${i} for cluster ${CLUSTER_NAME}..."
  sbatch sequential_job.sh ${i} ${CLUSTER_NAME}
}

# make the first request
i=1
request_job


while [[ (${#JOBS_WHICH_HAVE_STARTED_RUNNING[@]} -ne 5) || ($(currently_running_job) -ne "") ]]; do
  echo "JOBS_WHICH_HAVE_STARTED_RUNNING: ${JOBS_WHICH_HAVE_STARTED_RUNNING[@]}"
  echo "JOBS_WHICH_HAVE_FINISHED: ${JOBS_WHICH_HAVE_FINISHED[@]}"
  CURRENTLY_RUNNING_JOB=$(currently_running_job)
  echo "CURRENTLY_RUNNING_JOB: ${CURRENTLY_RUNNING_JOB}"

  # if CURRENTLY_RUNNING_JOB is not in JOBS_WHICH_HAVE_STARTED_RUNNING, add it
  if [[ ! " ${JOBS_WHICH_HAVE_STARTED_RUNNING[@]} " =~ " ${CURRENTLY_RUNNING_JOB} " ]]; then
    JOBS_WHICH_HAVE_STARTED_RUNNING+=(${CURRENTLY_RUNNING_JOB})
  fi

  # Iterate through each job j in JOBS_WHICH_HAVE_STARTED_RUNNING.
  # If one of the jobs j is not equal to CURRENTLY_RUNNING_JOB and is also not in JOBS_WHICH_HAVE_FINISHED, we will:
  # - add it to JOBS_WHICH_HAVE_FINISHED
  # - request the next job
  for job in ${JOBS_WHICH_HAVE_STARTED_RUNNING[@]}; do
    echo "job: ${job}"
    if [[ ${job} != ${CURRENTLY_RUNNING_JOB} ]] && [[ ! " ${JOBS_WHICH_HAVE_FINISHED[@]} " =~ " ${job} " ]]; then
      echo "job ${job} is finished"
      JOBS_WHICH_HAVE_FINISHED+=(${job})
      i=$((i + 1))
      request_job
    fi
  done

  sleep 10 # Optional: pause for 1 second
done

#sbatch sequential_job.sh 1 KILLARNEY
