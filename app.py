import os
import shutil
import streamlit as st

from collections import Counter
from utils.record import recorder
from utils.segments import segment, convert_to_wav
from utils.utils import (
    get_random_otp,
    read_code,
    write_code,
    load_model_tensor,
    load_model_pytorch,
    load_and_transform,
    load_decoder,
    extract_feature,
    predict_speaker
)


# const
WORKING_DIR = os.getcwd()
NUMS = [1, 2, 3, 4, 5]
NUM_OF_PASS_CHAR = 4
SECRET_CODE_PATH = os.path.join(
    os.path.expanduser('~'), 
    'secret_code.txt'
)
OUTPUT_WAVE_PATH = os.path.join(
    WORKING_DIR + '/audio/files',
    'record.wav'
)
OUTPUT_SEGMENT_PATH = os.path.join(
    WORKING_DIR,
    'audio/segments'
)
SAVE_FILE_PATH = os.path.join(
    WORKING_DIR + '/audio/files',
    'upload.wav'
)
MODEL_CLASSIFY_SPEAKER_PATH = os.path.join(
    WORKING_DIR,
    'models/weights/speaker_classification.h5'
)
DECODER_PATH = os.path.join(
    WORKING_DIR,
    'features/decoder.pkl'
)
SPEAKER_DECODER_PATH = os.path.join(
    WORKING_DIR,
    'features/speaker.pkl'
)
MODEL_CLASSIFY_NUMBER_PATH = os.path.join(
    WORKING_DIR,
    'models/weights/number_classification.ckpt'
)
loaded_model_pytorch = load_model_pytorch(MODEL_CLASSIFY_NUMBER_PATH)
model_classify_speaker = load_model_tensor(MODEL_CLASSIFY_SPEAKER_PATH)

def check_output_record():
    if os.path.exists(OUTPUT_WAVE_PATH):
        os.remove(OUTPUT_WAVE_PATH)

    if os.path.exists(SAVE_FILE_PATH):    
        os.remove(SAVE_FILE_PATH)

def check_before_segment():
    if os.path.exists(OUTPUT_SEGMENT_PATH):
        shutil.rmtree(OUTPUT_SEGMENT_PATH)
    os.makedirs(OUTPUT_SEGMENT_PATH)


def check_after_segment():
    if len(os.listdir(OUTPUT_SEGMENT_PATH)) == NUM_OF_PASS_CHAR:
        st.success('Segmented successfully!')
        return True
    else:
        st.error('Segmented failed. Retry!')
        return False


def call_model_number(filepath):
    specs = load_and_transform(filepath)
    specs = specs.unsqueeze(0).unsqueeze(0)
    label_predicted = loaded_model_pytorch(specs).item()
    return NUMS[label_predicted - 1]


def call_model_speaker(fname_segment):
    decoder = load_decoder(DECODER_PATH)
    speaker_decoder = load_decoder(SPEAKER_DECODER_PATH)

    fv = extract_feature(fname_segment)
    key_speaker = predict_speaker(
        model_classify_speaker,
        decoder,
        fv
    )
    
    speaker = speaker_decoder[key_speaker] if key_speaker in speaker_decoder else key_speaker
    return speaker


def get_speaker(dict_speaker):
    return sorted(dict_speaker.items(), key=lambda x: x[1], reverse=True)[0][0]


def call_model():
    tmp_nums, speaker = list(), list()
    otp_pass = read_code(SECRET_CODE_PATH)
                    
    for file in os.listdir(OUTPUT_SEGMENT_PATH):
        tmp_nums.append(call_model_number(os.path.join(OUTPUT_SEGMENT_PATH, file)))
        speaker.append(call_model_speaker(os.path.join(OUTPUT_SEGMENT_PATH, file)))

    pred_otp = ''.join(str(i) for i in tmp_nums)
    pred_speaker = get_speaker(dict(Counter(speaker)))

    if otp_pass == pred_otp and pred_speaker:
        st.text(f'Speaker: {pred_speaker}')
        st.text(f'OTP predict: {pred_otp}')
        st.success('You passed!')
    else:
        st.error("Can't authenticate. Try again!")

def main():
    st.title('Welcome to Voice OTP app')
    
    st.markdown('1. Click on button **Get OTP** to get your random password')
    btn_get_pass = st.button("Get OTP")

    if btn_get_pass:
        check_output_record()
        check_before_segment()

        otp_pass = get_random_otp(NUMS, k=NUM_OF_PASS_CHAR)
        write_code(otp_pass, SECRET_CODE_PATH)
        st.write('Your password is: ', otp_pass)
        st.info('Remember your password before you recording!')

    st.markdown('2. Click on button **Record** and read your password or **Upload file**')
    btn_record = st.button('Record')
    upload_file = st.file_uploader(
        'Choose your recorded file',
        accept_multiple_files=False,
        type=['wav']
    )

    st.markdown('3. Wait for the model to predict the result')
    
    if btn_record:
        check_output_record()
        check_before_segment()
        stt_record = False
        with st.spinner('Recording...'):
            st.info('Press q button on your keyboard to stop recording!')
            if recorder(fname=OUTPUT_WAVE_PATH):
                stt_record = segment(OUTPUT_WAVE_PATH, OUTPUT_SEGMENT_PATH)
            else:
                st.error('Recorded failed. Retry!')

        if check_after_segment() and stt_record:
            with st.spinner('Predicting...'):
                call_model()


    if upload_file:
        check_output_record()
        check_before_segment()

        convert_to_wav(upload_file, SAVE_FILE_PATH)
        segment(SAVE_FILE_PATH, OUTPUT_SEGMENT_PATH)
        if check_after_segment():
            with st.spinner('Predicting...'):
                call_model()


if __name__ == '__main__':
    main()
