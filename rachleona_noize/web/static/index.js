window.onload = () => {
    document.querySelectorAll("div.slider-wrapper").forEach(e => {
        e.querySelector('p.slider-value').innerText = e.querySelector('input.uk-range').value
        e.querySelector('input.uk-range').addEventListener('change', event => {

            e.querySelector('p.slider-value').innerText = event.target.value
        })
    })

    document.querySelector('#reset-btn').addEventListener('click', () => {
        document.querySelector('form').reset()
        document.querySelector('#pl-value').innerText = 5
        document.querySelector('#i-value').innerText = 300
    })

    document.querySelector('#submit-btn').addEventListener('click', () => {
        document.querySelector('form').submit()
    })

    document.querySelectorAll('.res-entry > form').forEach(e => {
        e.addEventListener('submit', downloadListener)
    })

    const pc = new ProgressChecker()
    const t = pc.startMonitoring()

    window.addEventListener('beforeunload', () => {
        clearInterval(t)
        pc.current = null
    });
}

const initCurrentJob = data => {

        const li = document.getElementById(data.job_id)
        li.querySelector('.uk-text-warning').setAttribute('hidden', 'hidden')
        const progressDiv = document.createElement('div')
        const segReport = document.createElement('p')
        segReport.innerHTML = `Processing segment <span class='cur-seg'>1</span> / ${ data.nseg }`
        progressBar.classList.add('uk-progress')
        progressBar.value = 0
        progressBar.max = data.iterations
        progressDiv.appendChild(segReport)
        progressDiv.appendChild(progressBar)
}

const transferCompletedJob = id => {
    document.getElementById(id).remove()
    const result = fetch(`/result/${ id }`).then(res => res.json()).then(json => json)
    
    const resNode = document.createElement('div')
    resNode.classList.add('uk-width-expand res-entry')
    res.innerHTML = `<h4><strong>${ result.output_filename }</strong></h4>
                    <ul class="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
                        <li><a href="#">Noise level ${ result.perturbation_level }</a></li>
                        <li><a href="#">${ result.iterations } iterations</a></li>
                        <li><a href="#">Target: ${ result.target }</a></li>
                        <li><a href="#">Encoders: ${ result.encoders }</a></li>
                    </ul>
                    <form class="uk-form-stacked" data-job="${ result.job_id }" data-output="${ result.output_filename }">
                        <div>
                            <label><input name="avc" class="uk-checkbox" type="checkbox" checked> Delete after download</label>
                        </div>
                        <button class="uk-margin uk-button uk-button-danger uk-width-1-1">Download</button>
                    </form>`

    const resDiv = document.getElementById('results-card').querySelector('uk-card-body')
    resDiv.insertBefore(resNode, resDiv.firstChild())
    resNode.querySelector('form').addEventListener('submit', downloadListener)

}

const downloadListener = event => {
    //todo : maybe notify if success?
    event.preventDefault()
    const form = event.target

    fetch(`/download/${ form.dataset.output }`)

    if (form.querySelector('.uk-checkbox').checked) {
        fetch(`/delete/${ form.dataset.job }`, { method: 'DELETE '})
    }
    
}

const checkObjectEmpty = obj => {
    res = true
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            res = false;
            break;
        }
    }

    return res
}

class ProgressChecker {
    constructor() {
      this.current = null;
    }

    pingProgress() {
        fetch('/progress').then(res => res.json()).then(json => {
            console.log(json)
            if (checkObjectEmpty(json)) {
                if (!this.current) return
                transferCompletedJob(this.current)
            }

            if (json.job_id !== this.current) {
                transferCompletedJob(this.current)
                this.current = json.job_id
                initCurrentJob(json)
            } else {
                const li = document.getElementById(json.job_id)
                li.querySelector('.cur-seg').innerText = json.current_seg
                li.querySelector('progress').value = json.current_progress
            }
        })
    }
    startMonitoring() {
        this.pingProgress()
        return setInterval(() => this.pingProgress(), 1000)
    }
  }