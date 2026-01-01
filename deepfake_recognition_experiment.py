from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice', 'pygame']

from psychopy import sound, core, visual, event, gui, data
import pandas as pd
import random
import os
import parselmouth
import numpy as np
from parselmouth.praat import call


print(f"Parselmouth version: {parselmouth.VERSION}")
print(f"Praat version: {parselmouth.PRAAT_VERSION}")
print(f"Praat version date: {parselmouth.PRAAT_VERSION_DATE}")

# GETTING PARTICIPANT INFORMATION

dialog = gui.Dlg(title = "Participant information")

dialog.addField("Participant ID: ")
dialog.addField("Age: ")
dialog.addField("Gender: ", choices=["Female", "Male", "Other"])
dialog.addField("Is English your native language?", choices = ["No", "Yes"])
dialog.addField("How familiar you are with deepfakes?", 
    choices = ["Completely unfamiliar", "Unfamiliar", "Somewhat familiar", "Familiar", "Very familiar"])

ok_data = dialog.show()

if ok_data is not None:
    ID = ok_data[0]
    age = ok_data[1]
    gender = ok_data[2]
    nativity = ok_data[3]
    familiarity = ok_data[4]
    
else:
    core.quit()




# GETTING THE AUDIO FILES

## Getting files for main experiment

fake_easy_folder = "stimuli/fake/easy"
fake_hard_folder = "stimuli/fake/hard"
real_easy_folder = "stimuli/real/easy"
real_hard_folder = "stimuli/real/hard"


real_easy_files = [os.path.join(real_easy_folder, f).replace('\\', '/') for f in os.listdir(real_easy_folder) 
              if f.endswith(('.flac'))]
real_hard_files = [os.path.join(real_hard_folder, f).replace('\\', '/') for f in os.listdir(real_hard_folder) 
              if f.endswith(('.flac'))]
fake_easy_files = [os.path.join(fake_easy_folder, f).replace('\\', '/') for f in os.listdir(fake_easy_folder) 
              if f.endswith(('.flac'))]
fake_hard_files = [os.path.join(fake_hard_folder, f).replace('\\', '/') for f in os.listdir(fake_hard_folder) 
              if f.endswith(('.flac'))]
              
print(f"Loaded {len(real_easy_files)} real_easy, {len(real_hard_files)} real_hard, and {len(fake_easy_files)} fake_easy and {len(fake_hard_files)} fake_hard audio files")
            
## Getting audio files for practice
practice_fake_folder = "stimuli/practice_fake"
practice_real_folder = "stimuli/practice_real"

practice_real_files = [os.path.join(practice_real_folder, f).replace('\\', '/') for f in os.listdir(practice_real_folder) if f.endswith(('.flac'))]
practice_fake_files = [os.path.join(practice_fake_folder, f).replace('\\', '/') for f in os.listdir(practice_fake_folder) if f.endswith(('.flac'))]

print(f"Loaded {len(practice_real_files)} practice real and {len(practice_fake_files)} practice fake audio files")


# CREATING TRIAL LIST
trials = []

## Adding real trials
for filepath in real_easy_files:
    trials.append({
        'filepath': filepath,
        'ActualAuthenticity': 'real',
        'Difficulty': 'easy',
        'filename': os.path.basename(filepath)
    })
    
for filepath in real_hard_files:
    trials.append({
        'filepath': filepath,
        'ActualAuthenticity': 'real',
        'Difficulty': 'hard',
        'filename': os.path.basename(filepath)
    })

## Adding fake trials  
for filepath in fake_easy_files:
    trials.append({
        'filepath': filepath,
        'ActualAuthenticity': 'fake',
        'Difficulty': 'easy',
        'filename': os.path.basename(filepath)
    })
    
for filepath in fake_hard_files:
    trials.append({
        'filepath': filepath,
        'ActualAuthenticity': 'fake',
        'Difficulty': 'hard',
        'filename': os.path.basename(filepath)
    })

## Randomizing trial order

trials = []

for f in real_easy_files:
    trials.append({"filepath": f, "Condition": "real_easy", "ActualAuthenticity": "real", "Difficulty": "easy", 'filename': os.path.basename(f)})

for f in real_hard_files:
    trials.append({"filepath": f, "Condition": "real_hard", "ActualAuthenticity": "real", "Difficulty": "hard", 'filename': os.path.basename(f)})

for f in fake_easy_files:
    trials.append({"filepath": f, "Condition": "fake_easy", "ActualAuthenticity": "fake", "Difficulty": "easy", 'filename': os.path.basename(f)})

for f in fake_hard_files:
    trials.append({"filepath": f, "Condition": "fake_hard", "ActualAuthenticity": "fake", "Difficulty": "hard", 'filename': os.path.basename(f)})

random.shuffle(trials)

print(f"Total trials: {len(trials)}")


# EXTRACTING ACOUSTIC FEATURES FROM PRAAT

def extract_acoustic_features(filepath):
    snd = parselmouth.Sound(filepath)

    # Pitch (F0)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]  # remove unvoiced frames
    f0_mean = np.mean(f0_values) if len(f0_values) > 0 else np.nan
    f0_std = np.std(f0_values) if len(f0_values) > 0 else np.nan

    # Intensity
    intensity = snd.to_intensity()
    int_values = intensity.values[0]
    intensity_mean = np.mean(int_values)
    intensity_std = np.std(int_values)
    
    # Formants (F1, F2, F3 at midpoint)
    formants = snd.to_formant_burg()
    duration = snd.duration
    midpoint = duration / 2
    F1 = formants.get_value_at_time(1, midpoint) or np.nan
    F2 = formants.get_value_at_time(2, midpoint) or np.nan
    F3 = formants.get_value_at_time(3, midpoint) or np.nan

    # Jitter (local)
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 75, 500, 1.3)

    # Shimmer (local)
    shimmer = parselmouth.praat.call(
        [snd, point_process],
        "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6
    )

    # Harmonics-to-noise ratio (HNR)
    hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_mean = parselmouth.praat.call(hnr, "Get mean", 0, 0)

    # Spectral features
    spectrum = snd.to_spectrum()
    freqs = spectrum.xs()
    power = spectrum.ys()[0]
    
    spectral_centroid = np.sum(freqs * power) / np.sum(power)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * power) / np.sum(power))
    spectral_rolloff = freqs[np.where(np.cumsum(power) >= 0.85 * np.sum(power))[0][0]]
    zcr = ((snd.values[0][:-1] * snd.values[0][1:]) < 0).sum() / snd.duration
    
    return {
        "F0_mean": f0_mean,
        "F0_std": f0_std,
        "Intensity_mean": intensity_mean,
        "Intensity_std": intensity_std,
        "F1": F1,
        "F2": F2,
        "F3": F3,
        "Jitter": jitter,
        "Shimmer": shimmer,
        "HNR": hnr_mean,
        "SpectralCentroid": spectral_centroid,
        "SpectralBandwidth": spectral_bandwidth,
        "SpectralRolloff": spectral_rolloff,
        "ZeroCrossingRate": zcr
    }

# CREATING THE EXPERIMENT

win = visual.Window(color = "black", fullscr = True)

def show_message(win, text, height = 0.1, wait = None, alignText = 'center'):
    m = visual.TextStim(win, text, height = height, alignText = alignText, anchorHoriz='center')
    m.draw()
    win.flip()
    if wait:
        core.wait(wait)

## INTRODUCTION

show_message(win, "Welcome to the experiment!", wait = 2)

show_message(win,
    """CONSENT FORM
    You are being invited to take part in a research study for our Perception and Action exam.
    The study investigates how people perceive and discriminate audio deepfakes from human recordings based on different acoustic features.
    For more information, you can contact the researchers.
    
    Inclusion Criteria:
        - 18 years of age or older
        - Normal or corrected vision
        - Does not have phonagnosia or any other impairment of voice recognition
    
    Principal Investigator(s) and Contact Information (mail):
        - Aurelija Spunde, 202405122@post.au.dk
        - Hannah Cohen, 202410037@post.au.dk
    
    Benefits and Risks
    There will be no risks other than those you meet in daily life. You will not receive any financial compensation, but you will be able to get some sweet treats.
    
    Press 'SPACE' to continue""", height = 0.05, alignText = 'left'
)
key = event.waitKeys(keyList=['space', 'escape'])
 
if key[0][0] == 'escape':
    win.close()
    core.quit()

show_message(win, 
    """Study Procedure
    You will be presented with short audio recordings,
    after which you will have to indicate whether you think the recording was fake (AI-made) or real (human voice).
    After that you will have to evaluate how confident you are in your judgment on a scale from 1-5 and how natural you think the recording sounded (on a scale from 1-5). 
    Before the actual experiment will start, you will have a short practice trial to familiarize yourself with the procedure.
    
    Confidentiality and Data Handling
    In this study, we will collect data about your age, gender (for demographic purposes) and information whether English is your native language (for research purposes).
    The data will be anonymous right after you finish the experiment, as you will be given a unique participant number.
    You can withdraw your consent and stop the experiment at any point by pressing ‘ESC' before the end of the experiment.
    Right after the experiment your data can still be retracted and deleted, if you wish so, by using your participant ID, until the day after the experiment.
    The data will be stored in a secure manner and deleted after the 5th of January.
    
    Participant consent statement
    By pressing 'SPACE', I state that I understand the terms and conditions that were presented and that I am allowed to withdraw my consent at any point before the day after the experiment. 
    I hereby give my consent to be the subject of the research and that my data can be used as described in this document.

    Press 'SPACE' to continue or 'ESC' to exit""", height = 0.05, alignText = 'left'
)
key = event.waitKeys(keyList=['space', 'escape'])
 
if key[0][0] == 'escape':
    win.close()
    core.quit()

show_message(win, "You will hear short audio clips.\n\n " \
    "Press 'R' if you think it is REAL" \
    "\n Press 'F' if you think it is FAKE" \
    "\n\nThen you will rate how confident you are in your answer (1-5)"
    "\n And how natural did it sound (1-5)"
    "\n\n Press 'SPACE' to continue"
)
key = event.waitKeys(keyList=['space', 'escape'])
 
if key[0][0] == 'escape':
    win.close()
    core.quit()

# PRACTICE BLOCK

## Creating practice trials
practice_trials = []

for filepath in practice_fake_files:
    practice_trials.append({
        'filepath': filepath,
        'ActualAuthenticity': 'fake',
        'filename': os.path.basename(filepath)
    })
    
for filepath in practice_real_files:
    practice_trials.append({
        'filepath': filepath,
        'ActualAuthenticity': 'real',
        'filename': os.path.basename(filepath)
    })

random.shuffle(practice_trials)

print(f"Total practice trials: {len(practice_trials)}")

## Running practice

show_message(win, text = "You will now complete a short PRACTICE.\n\nPress SPACE to begin.")
event.waitKeys(keyList=["space"])

for trial in practice_trials:
    # fixation
    show_message(win, text="+", height=0.18)

    audio = sound.Sound(trial["filepath"])
    audio.play()
    core.wait(audio.getDuration())
    
    win.flip()
    core.wait(0.3)

    show_message(win, text = "Real (R) or Fake (F)?")
    clock = core.Clock()
    keys = event.waitKeys(keyList=["r", "f", "escape"], timeStamped=clock)
    
    if keys[0][0] == 'escape':
        break
    
    # Get confidence rating
    show_message(win,
    text="How confident are you? \n\n1=Not at all \n5=Very confident\nPress number 1-5",
    height=0.1)
    conf_keys = event.waitKeys(keyList=['1', '2', '3', '4', '5', 'escape'])
    
    if conf_keys[0] == 'escape':
        break
        
    # Get naturalness rating
    show_message(win,
    text="How natural did it sound?\n\n1=Very unnatural \n5=Very natural\nPress number 1-5",
    height=0.1)
    nat_keys = event.waitKeys(keyList=['1', '2', '3', '4', '5', 'escape'])
    
    if nat_keys[0] == 'escape':
        break

show_message(win, text = "Practice complete!\n\nPress SPACE to begin the main experiment.")

key = event.waitKeys(keyList=['space', 'escape'])
 
if key[0][0] == 'escape':
    win.close()
    core.quit()


# RUNNING THE MAIN EXPERIMENT

trial_data = []

for trial_num, trial in enumerate(trials):
    # Fixation
    show_message(win, text="+", height=0.2)
    
    # Play audio
    audio = sound.Sound(trial['filepath'])
    audio.play()

    # Wait for audio to finish
    core.wait(audio.getDuration())
    
    # Small pause after audio
    win.flip()
    core.wait(0.3)
    
    # Get response
    show_message(win, text="Real (R) or Fake (F)?", height=0.1)

    # Wait for R or F key
    clock = core.Clock()
    keys = event.waitKeys(keyList=['r', 'f', 'escape'], timeStamped=clock)
    
    if keys[0][0] == 'escape':
        break
    
    response = keys[0][0]
    rt = keys[0][1]
    
    # Get confidence rating
    show_message(win,
    text="How confident are you? \n\n1=Not at all \n5=Very confident\nPress number 1-5",
    height=0.1)
    conf_keys = event.waitKeys(keyList=['1', '2', '3', '4', '5', 'escape'])
    
    if conf_keys[0] == 'escape':
        break
        
    confidence = int(conf_keys[0])

    # Get naturalness rating
    show_message(win,
    text="How natural did it sound?\n\n1=Very unnatural \n5=Very natural\nPress number 1-5",
    height=0.1)
    nat_keys = event.waitKeys(keyList=['1', '2', '3', '4', '5', 'escape'])
    
    if nat_keys[0] == 'escape':
        break
        
    naturalness = int(nat_keys[0])

    # Calculate accuracy
    
    correct = 1 if (
        (response == 'r' and trial['ActualAuthenticity'] == 'real') or \
        (response == 'f' and trial['ActualAuthenticity'] == 'fake')
    ) else 0
    
    # Acoustic features
    acoustics = extract_acoustic_features(trial["filepath"])
    
    # Store data
    trial_data.append({
        "ParticipantID": ID, 
        "Age": age, 
        "Gender": gender, 
        "EnglishNativity": nativity, 
        "Familiarity": familiarity,
        "Trial": trial_num +1,
        "Filename": trial['filename'], 
        "ResponseTime": rt, 
        "ActualAuthenticity": trial['ActualAuthenticity'],
        "Difficulty": trial['Difficulty'],
        "Condition": trial['Condition'],
        "Response": 'real' if response == 'r' else 'fake',
        "Correct": correct, 
        "Confidence": confidence, 
        "Naturalness": naturalness,
        **acoustics
    })
    
    print(f"Trial {trial_num + 1}/{len(trials)} completed")
    
    # Brief inter-trial interval
    core.wait(0.5)



# ENDING

logfile = pd.DataFrame(trial_data)

accuracy = logfile["Correct"].mean() * 100
num_correct = logfile["Correct"].sum()

show_message(win,
    text = f"Thank you for participating! \n\nYou answered correctly on {num_correct} out of {len(logfile)} trials.\n"
    "Press any key to exit"
)
event.waitKeys()

os.makedirs("logfiles", exist_ok=True)

date = data.getDateStr().replace(":", "-").replace(" ", "_")

logfile_name = f"logfiles/logfile_{ID}_{date}.csv"
logfile.to_csv(logfile_name, index= False)

print(f"\nData saved! Total trials: {len(logfile)}")
print(f"Accuracy: {logfile['Correct'].mean()*100:.1f}%")

win.close()
core.quit()