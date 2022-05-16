import os
import shutil
import random
import streamlit as st

from utils.record import recorder
from utils.segments import segment, convert_to_wav
from utils.utils import (
    load_model,
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


def random_otp(k):
    otp_pass = 0
    for num in random.sample(NUMS, k=k):
        otp_pass = otp_pass * 10 + num
    
    return otp_pass


def write_code(code):
    with open(SECRET_CODE_PATH, 'w') as file:
        file.write(str(code))
        file.close()


def read_code():
    with open(SECRET_CODE_PATH, 'r') as file:
        code = int(file.read())
        file.close()

    return code


def check_before_segment():
    if os.path.exists(OUTPUT_SEGMENT_PATH):
        shutil.rmtree(OUTPUT_SEGMENT_PATH)
        os.makedirs(OUTPUT_SEGMENT_PATH)


def check_after_segment():
    if len(os.listdir(OUTPUT_SEGMENT_PATH)) == 1:
        st.success('Segmented successfully!')
        return True
    else:
        st.error('Segmented failed. Retry!')
        return False


def main():
    st.title('Welcome to Voice OTP app!')
    
    st.markdown('1. Click on button **Get OTP** to get your random password')
    btn_get_pass = st.button("Get OTP")

    if btn_get_pass:
        otp_pass = random_otp(k=4)
        write_code(otp_pass)
        st.write('Your password is: ', otp_pass)
        st.info('Remember your password before you recording!')

    st.markdown('2. Click on button **Record** and read your password or **Upload file**')
    upload_file = st.file_uploader(
        'Choose your recorded file',
        accept_multiple_files=False,
        type=['wav', 'flv', 'ogg', 'mp3']
    )
    btn_record = st.button('Record')

    if upload_file:
        check_before_segment()
        convert_to_wav(upload_file, SAVE_FILE_PATH)
        segment(SAVE_FILE_PATH, OUTPUT_SEGMENT_PATH)
        if check_after_segment():
            st.markdown('3. Wait for the model to predict the result')
            with st.spinner('Predicting...'):
                model_classify_speaker = load_model(MODEL_CLASSIFY_SPEAKER_PATH)
                decoder = load_decoder(DECODER_PATH)

                fv = extract_feature(os.path.join(OUTPUT_SEGMENT_PATH, 'segment_0.wav'))
                speaker = predict_speaker(
                    model_classify_speaker,
                    decoder,
                    fv
                )
                
                if speaker:
                    st.text(f'Speaker: {speaker}')


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
                    st.markdown('3. Wait for the model to predict the result')
                    with st.spinner('Predicting...'):
                        model_classify_speaker = load_model(MODEL_CLASSIFY_SPEAKER_PATH)
                        decoder = load_decoder(DECODER_PATH)

                        fv = extract_feature(os.path.join(OUTPUT_SEGMENT_PATH, 'segment_0.wav'))
                        speaker = predict_speaker(
                            model_classify_speaker,
                            decoder,
                            fv
                        )
                        
                        if speaker:
                            st.text(f'Name: {speaker}')
            else:
                st.error('Recorded failed. Retry!')
    

if __name__ == '__main__':
    main()