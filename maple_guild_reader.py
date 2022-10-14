from encodings import utf_8
import cv2
import requests
import os

LIMIT_PX = 1024
LIMIT_BYTE = 1024*1024  # 1MB
LIMIT_BOX = 40


def kakao_ocr_resize(image_path: str):
    """
    ocr detect/recognize api helper
    ocr api의 제약사항이 넘어서는 이미지는 요청 이전에 전처리가 필요.

    pixel 제약사항 초과: resize
    용량 제약사항 초과  : 다른 포맷으로 압축, 이미지 분할 등의 처리 필요. (예제에서 제공하지 않음)

    :param image_path: 이미지파일 경로
    :return:
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    if LIMIT_PX < height or LIMIT_PX < width:
        ratio = float(LIMIT_PX) / max(height, width)
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        height, width, _ = height, width, _ = image.shape

        # api 사용전에 이미지가 resize된 경우, recognize시 resize된 결과를 사용해야함.
        image_path = "{}_resized.jpg".format(image_path)
        cv2.imwrite(image_path, image)

        return image_path
    return None


def kakao_ocr(image_path: str, appkey: str):
    """
    OCR api request example
    :param image_path: 이미지파일 경로
    :param appkey: 카카오 앱 REST API 키
    """
    API_URL = 'https://dapi.kakao.com/v2/vision/text/ocr'

    headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

    image = cv2.imread(image_path)
    jpeg_image = cv2.imencode(".jpg", image)[1]
    data = jpeg_image.tobytes()


    return requests.post(API_URL, headers=headers, files={"image": data})

def find_image_in_folder(image_folder_path : str):
    root_dir = image_folder_path
    
    img_path_list = []
    possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png'] # 이미지 확장자들
    
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    img_path = root + '/' + file_name
                    
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
                    img_path_list.append(img_path)
                                
    return img_path_list


def preprocess_image(image_path : str):
    img = cv2.imread(image_path)
    width = len(img[0])
    height = len(img)
    start_width = int(width/ 3) + 130
    end_width = int(width / 3 * 2) + 130
    start_height = int(height / 3) - 100
    end_height = int(height / 3 * 2) + 50

    cropped_img = img[start_height:end_height, start_width:end_width]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    ret, thred_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

    processed_img = thred_img
    processed_img[:,83:170] = 255

    thred_img_path = image_path.replace(".jpg","_thred.jpg")
    thred_img_path = image_path.replace("data","temp")

    if not os.path.exists("./temp"):
        os.makedirs("./temp")

    cv2.imwrite(thred_img_path, processed_img)

    return thred_img_path

def sort_maple_gulid_score(ocr_result : dict):

    key_list = ['캐릭터명', '레벨', '직위', '주간점수', '수로점수', '플래그']


    data_dict = {}
    for key in key_list:
        data_dict[key] = []

    pre_y = 4

    for data in ocr_result['result']:
        #data = output['result'][0]
        x = data['boxes'][0][0]
        y = data['boxes'][0][1]
        word = data['recognition_words'][0]

        # 다른 캐릭터
        if y - pre_y > 10:
            total_count = len(data_dict['캐릭터명'])
            for value in data_dict.values():
                if len(value) < total_count:
                    value.append("?")

        pre_y = y

        if x >= 10 and x <= 20:
            # 이름
            data_dict['캐릭터명'].append(word)
            continue
        elif x >= 165 and x <= 175:
            # 레벨
            data_dict['레벨'].append(word)
            continue
        elif x >= 215 and x <= 225:
            # 등긃
            data_dict['직위'].append(word)
            continue
        elif x >= 280 and x <= 290:
            # 주간 미션 포인트
            data_dict['주간점수'].append(word)
            continue
        elif x >= 330 and x <= 350:
            # 수로 점수
            data_dict['수로점수'].append(word)
            continue
        elif x >= 415 and x <= 430:
            data_dict['플래그'].append(word)
            # 플래그 점수
            continue
    
    total_count = len(data_dict['캐릭터명'])
    for value in data_dict.values():
        if len(value) < total_count:
            value.append("?")

    return data_dict


def change_dict_to_list(result : dict):
    nick_list = result['캐릭터명']
    level_list = result['레벨']
    grade_list = result['직위']
    week_point_list = result['주간점수']
    suro_point_list = result['수로점수']
    flag_point_list = result['플래그']

    result_list = []

    # 닉네임체크
    wrong_dict = None
    if os.path.exists("./wrong_word.txt"):
        print("오타 교정을 합니다")
        file = open("./wrong_word.txt",'rt', encoding='UTF8')
        wrong_dict = dict()
        while True:
            line = file.readline()
            if not line:
                break;
            data = line.split("\t")
            wrong_dict[data[0]] = data[1].strip()
        file.close()
    else:
        print("오타 교정 파일이 없습니다. 오타 교정을 하지 않습니다.")

    for i in range(len(nick_list)):
        temp = []
        if wrong_dict is None:
            temp.append(nick_list[i])
        else:
            if nick_list[i] in wrong_dict:
                temp.append(wrong_dict[nick_list[i]])
            else:
                temp.append(nick_list[i])
        temp.append(level_list[i])
        temp.append(grade_list[i])
        temp.append(week_point_list[i])
        temp.append(suro_point_list[i])
        temp.append(flag_point_list[i])
        result_list.append(temp)

    return result_list

def main():
    # if len(sys.argv) != 3:
    #     print("Please run with args: $ python example.py /path/to/image appkey")
    appkey_file_path = "./appkey.txt"
    image_folder_path = "./data"

    appkey_file = open(appkey_file_path)
    appkey = appkey_file.readline()

    images_path = find_image_in_folder(image_folder_path)

    result_file_path = "./result.txt"

    result_file = open(result_file_path,'+w', encoding="utf_8")

    for image_path in images_path:
        print(image_path + "시작")
        processed_image_path = preprocess_image(image_path)
        ocr_result = kakao_ocr(processed_image_path, appkey).json()
        os.remove(processed_image_path)
        result = sort_maple_gulid_score(ocr_result)
        result_list = change_dict_to_list(result)
        print(image_path + "완료" + "\n")
        for result_text in result_list:
            for value in result_text:
                result_file.write(str(value) + "\t")
            result_file.write("\n")
    
    result_file.close()

    if os.path.isdir("./temp"):
        os.removedirs("./temp")


if __name__ == "__main__":
    output = main()