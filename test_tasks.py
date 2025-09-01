from db_tasks import create_task, fetch_and_claim_next_task, complete_task, fail_task

# create a task
tid = create_task(requester="cli-user", task_type="fetch_weather", payload={"lat":12.97,"lon":77.59})
print("Created task id:", tid)

# claim
task = fetch_and_claim_next_task("test-agent-1")
print("Claimed:", task)

# complete (if claimed)
if task:
    ok = complete_task(task["id"], {"status":"done","note":"okay"})
    print("Completed:", ok)
