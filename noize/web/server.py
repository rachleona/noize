import os

from flask import Flask, render_template, request, send_file
import noize.web.queue as q

app = Flask(__name__)
dirpath = os.path.dirname(os.path.dirname(__file__))
voices_path = os.path.join(dirpath, "misc", "voices")
voices = os.listdir(voices_path)


@app.route("/")
def main():
    current_tasks = [] if q.in_progress is None else [q.in_progress.get_metadata()]
    for i in range(len(q.task_queue)):
        current_tasks.append(q.task_queue[i].get_metadata())
    current_results = [q.results[x].get_metadata() for x in q.results]
    return render_template(
        "index.html", tasks=current_tasks, done=current_results, voices=voices
    )


@app.route("/", methods=["POST"])
def upload_and_protect():
    f = request.files["src_audio"]
    new_job = q.ProtectJob(
        f,
        request.form["perturbation_level"],
        request.form["target"],
        request.form.get("avc", False),
        request.form.get("freevc", False),
        request.form.get("xtts", False),
        request.form["iterations"],
    )
    q.task_queue.append(new_job)

    return main()


@app.route("/download/<filename>")
def download(filename=None):
    return send_file(os.path.join(q.outputs_path, filename), as_attachment=True)


@app.route("/<job_id>", methods=["DELETE"])
def delete_result(job_id=None):
    if job_id is None:
        return 400

    res = q.results[job_id]
    os.remove(os.path.join(q.outputs_path, res.job_id.hex + ".wav"))
    del q.results[job_id]

    return 200


@app.route("/progress")
def peek_progress():
    return q.in_progress.get_progress() if q.in_progress is not None else {}


@app.route("/result/<job_id>")
def get_result(job_id=None):
    if job_id is None or job_id not in q.results:
        return {}

    return q.results[job_id].get_metadata()
