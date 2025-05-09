# `Noize`

**Usage**:

```console
$ Noize [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `protect`: Apply adversarial perturbation to produce...
* `web`: Starts Flask server for web interface
* `voices`: Manage saved target voices to be used in...

## `Noize protect`

Apply adversarial perturbation to produce protected audio file

**Usage**:

```console
$ Noize protect [OPTIONS] [FILEPATH] [OUTPUT_DIR]
```

**Arguments**:

* `[FILEPATH]`: path to the audio file to apply protection on
* `[OUTPUT_DIR]`: name of output directory, will be created if not already exists

**Options**:

* `--perturbation-level INTEGER`: values between 1-10, controls how strong the noise applied is  [default: 5]
* `--target TEXT`: id of a saved voice for target-based optimisation
* `--config-file PATH`: path to the config file, use official repo one by default  [default: /home/rachleona/gmtfo/rachleona-noize/rachleona_noize/config.json]
* `--output-filename TEXT`: name of protected file, uses protected_&lt;filpath&gt; by default
* `--avc / --no-avc`: whether to use adaptVC encoder in perturbation calculation  [default: avc]
* `--freevc / --no-freevc`: whether to use FreeVC encoder in perturbation calculation  [default: freevc]
* `--xtts / --no-xtts`: whether to use XTTS encoder in perturbation calculation  [default: xtts]
* `--logs / --no-logs`: whether to log intermediate loss values  [default: no-logs]
* `--log-file TEXT`: name of log file is logs are requested  [default: log.csv]
* `--resource-log PATH`: name of resource log file if logs are requested
* `--learning-rate FLOAT`: learning rate to use for optimisation process  [default: 0.02]
* `--iterations INTEGER`: optimisation iterations (per audio segment)  [default: 300]
* `--distance-weight FLOAT`: weight used in loss functions  [default: 2]
* `--snr-weight FLOAT`: weight used in loss functions  [default: 0.025]
* `--perturbation-norm-weight FLOAT`: weight used in loss functions  [default: 0.25]
* `--frequency-weight FLOAT`: weight used in loss functions  [default: 1.5]
* `--avc-weight FLOAT`: weight used in loss functions  [default: 25]
* `--freevc-weight FLOAT`: weight used in loss functions  [default: 25]
* `--xtts-weight FLOAT`: weight used in loss functions  [default: 25]
* `--help`: Show this message and exit.

## `Noize web`

Starts Flask server for web interface

**Usage**:

```console
$ Noize web [OPTIONS]
```

**Options**:

* `--port INTEGER`: port for server to listen  [default: 5000]
* `--debug / --no-debug`: whether to start server in debug mode  [default: no-debug]
* `--help`: Show this message and exit.

## `Noize voices`

Manage saved target voices to be used in protection application

**Usage**:

```console
$ Noize voices [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: Lists the ids of all saved voices that can...
* `play`: Plays 10s of the reference clip given for...
* `add`: Add a new saved voice for use as...
* `reset`: Recalculates embeddings for a saved voice
* `resetall`: Recalculates embeddings for all saved voices
* `delete`: Deletes a saved voice from voice library

### `Noize voices list`

Lists the ids of all saved voices that can be used as perturbation targets, with descriptions

**Usage**:

```console
$ Noize voices list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `Noize voices play`

Plays 10s of the reference clip given for generating the voice embeddings of given id

**Usage**:

```console
$ Noize voices play [OPTIONS] ID
```

**Arguments**:

* `ID`: id of the saved voice to play, warning will be given if id does not exist  [required]

**Options**:

* `--help`: Show this message and exit.

### `Noize voices add`

Add a new saved voice for use as perturbation calculation target

**Usage**:

```console
$ Noize voices add [OPTIONS] ID [FILEPATH] [DESC]
```

**Arguments**:

* `ID`: id/name for new voice  [required]
* `[FILEPATH]`: path to the WAV file to be used as reference for new voice
* `[DESC]`: description string to be saved alongside the embeddings

**Options**:

* `--config-file PATH`: path to the config file containing metadata and hyperparameters needed for voice models used  [default: /home/rachleona/gmtfo/rachleona-noize/rachleona_noize/config.json]
* `--device TEXT`: device to conduct tensor calculations on  [default: cpu]
* `--help`: Show this message and exit.

### `Noize voices reset`

Recalculates embeddings for a saved voice

**Usage**:

```console
$ Noize voices reset [OPTIONS] ID
```

**Arguments**:

* `ID`: id of the saved voice to play, warning will be given if id does not exist  [required]

**Options**:

* `--config-file PATH`: path to the config file containing metadata and hyperparameters needed for voice models used  [default: /home/rachleona/gmtfo/rachleona-noize/rachleona_noize/config.json]
* `--device TEXT`: device to conduct tensor calculations on  [default: cpu]
* `--help`: Show this message and exit.

### `Noize voices resetall`

Recalculates embeddings for all saved voices

**Usage**:

```console
$ Noize voices resetall [OPTIONS]
```

**Options**:

* `--config-file PATH`: path to the config file containing metadata and hyperparameters needed for voice models used  [default: /home/rachleona/gmtfo/rachleona-noize/rachleona_noize/config.json]
* `--device TEXT`: device to conduct tensor calculations on  [default: cpu]
* `--help`: Show this message and exit.

### `Noize voices delete`

Deletes a saved voice from voice library

**Usage**:

```console
$ Noize voices delete [OPTIONS] ID
```

**Arguments**:

* `ID`: id of the saved voice to play, warning will be given if id does not exist  [required]

**Options**:

* `--help`: Show this message and exit.
