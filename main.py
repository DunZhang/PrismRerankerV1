from mteb import get_benchmark

tasks = get_benchmark("MTEB(eng, v2)")
for task in tasks:
    if task.metadata.type == "Retrieval":
        print(task.metadata.dataset.get("path"))
tasks = get_benchmark("MTEB(cmn, v1)")
for task in tasks:
    if task.metadata.type == "Retrieval":
        print(task.metadata.dataset.get("path"))