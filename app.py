import os
import shutil
import random
import operator
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
    'models/speaker_classification.h5'
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
    'models/number_classification.ckpt'
)


def check_before_segment():
    if os.path.exists(OUTPUT_SEGMENT_PATH):
        shutil.rmtree(OUTPUT_SEGMENT_PATH)
    os.makedirs(OUTPUT_SEGMENT_PATH)


def check_after_segment():
    if len(os.listdir(OUTPUT_SEGMENT_PATH)) == 4:
        st.success('Segmented successfully!')
        return True
    else:
        st.error('Segmented failed. Retry!')
        return False


def call_model_speaker(fname_segment):
    model_classify_speaker = load_model_tensor(MODEL_CLASSIFY_SPEAKER_PATH)
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


def call_model_number(filepath):
    loaded_model = load_model_pytorch(MODEL_CLASSIFY_NUMBER_PATH)
    specs = load_and_transform(filepath)
    specs = specs.unsqueeze(0).unsqueeze(0)
    label_predicted = loaded_model(specs).item()
    return NUMS[label_predicted - 1]


def check_otp():
    tmp_nums = list()
    otp_pass = read_code(SECRET_CODE_PATH)
    for file in os.listdir(OUTPUT_SEGMENT_PATH):
        tmp_nums.append(call_model_number(os.path.join(OUTPUT_SEGMENT_PATH, file)))
    
    pred_otp = ''.join(str(i) for i in tmp_nums)
    st.text(f'OTP predict: {pred_otp}')

    if otp_pass == pred_otp:
        return True
    
    return False


def main():
    st.title('Welcome to Voice OTP app!')
    
    st.markdown('1. Click on button **Get OTP** to get your random password')
    btn_get_pass = st.button("Get OTP")

    if btn_get_pass:
        otp_pass = get_random_otp(NUMS, k=4)
        write_code(otp_pass, SECRET_CODE_PATH)
        st.write('Your password is: ', otp_pass)
        st.info('Remember your password before you recording!')

    st.markdown('2. Click on button **Record** and read your password or **Upload file**')
    upload_file = st.file_uploader(
        'Choose your recorded file',
        accept_multiple_files=False,
        type=['wav']
    )
    btn_record = st.button('Record')

    st.markdown('3. Wait for the model to predict the result')

    if upload_file:
        check_before_segment()
        convert_to_wav(upload_file, SAVE_FILE_PATH)
        segment(SAVE_FILE_PATH, OUTPUT_SEGMENT_PATH)
        if check_after_segment():
            with st.spinner('Predicting...'):
                speaker = list()
                for file in os.listdir(OUTPUT_SEGMENT_PATH):
                    speaker.append(call_model_speaker(os.path.join(OUTPUT_SEGMENT_PATH, file)))
                
                if speaker:
                    st.text(f'Speaker: {get_speaker(dict(Counter(speaker)))}')
                
                if check_otp():
                    st.success('You passed!')
                    st.balloons()
                

    if btn_record:
        with st.spinner('Recording...'):
            st.info('Press q button on your keyboard to stop recording!')
            
            if os.path.exists(OUTPUT_WAVE_PATH):
                os.remove(OUTPUT_WAVE_PATH)

            status = recorder(fname=OUTPUT_WAVE_PATH)

            if status:
                check_before_segment()
                segment(OUTPUT_WAVE_PATH, OUTPUT_SEGMENT_PATH)
                
                if check_after_segment():
                    with st.spinner('Predicting...'):
                        speaker = list()
                        for file in os.listdir(OUTPUT_SEGMENT_PATH):
                            speaker.append(call_model_speaker(os.path.join(OUTPUT_SEGMENT_PATH, file)))
                        
                        if speaker:
                            st.text(f'Speaker: {get_speaker(dict(Counter(speaker)))}')
                        
                        if check_otp():
                            st.success('You passed!')
                            st.balloons()
            else:
                st.error('Recorded failed. Retry!')


if __name__ == '__main__':
    main()
