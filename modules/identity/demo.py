import cv2
import numpy as np
from visdom import Visdom

from modules.identity import PersonReIDImg_FeatureComparison

def scale_resize(img, img_size):
    h, w, _ = img.shape
    if h/w > img_size[0]/img_size[1]:
        ratio = img_size[0]/h
        new_h = img_size[0]
        new_w = int(ratio*w)
        bound_w = (img_size[1]-new_w)/2
        bound = (0, 64, int(bound_w), int(bound_w+0.5))
    else:
        ratio = img_size[1]/w
        new_h = int(ratio*h)
        new_w = img_size[1]
        bound_h = (img_size[0]-new_h)/2
        bound = (int(bound_h), int(bound_h+0.5)+64, 0, 0)
    new_size = (new_h, new_w)
    img = cv2.resize(img, (new_size[1], new_size[0]))
    img = cv2.copyMakeBorder(img,*bound,cv2.BORDER_CONSTANT, value=(0,0,0))
    return img

def demo(test_img_path, sample_path, n_p=20,n_n=10, show_img_size=(384, 192)):
    test_imgs = []
    test_img_names = os.listdir(test_img_path)[32:64]
    for img_name in test_img_names:
        img = cv2.imread(os.path.join(test_img_path, img_name))
        test_imgs.append(img)

    personreid_img_feature_comparison = PersonReIDImg_FeatureComparison(sample_path)
    res = personreid_img_feature_comparison.run(test_imgs)

    print(res.keys())
    for i in range(len(test_imgs)):
        print(res['score'][i])
        print(res['person_name'][i])
        print(res['img_name'][i])
        print('#'*64)


    imgs = []
    for test_img, test_img_name in zip(test_imgs, test_img_names):
        img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        img = scale_resize(img, show_img_size)

        # 在图片上添加name
        name = test_img_name.split('.')[0]
        text_color = (255, 255, 255)
        t_size_name = cv2.getTextSize(name, 0, fontScale=0.7, thickness=1)[0]
        text_pos_label = (int(img.shape[1]/2)-int(t_size_name[0]/2), img.shape[0]-2-t_size_name[1])
        img = cv2.putText(img, name, text_pos_label, 0, 0.7, text_color, thickness=1, lineType=cv2.LINE_AA)

        # 在图片上添加‘query'
        title = 'query'
        text_color = (0, 255, 0)
        t_size_title = cv2.getTextSize(title, 0, fontScale=0.7, thickness=1)[0]
        text_pos_title = (int(img.shape[1]/2)-int(t_size_title[0]/2), img.shape[0]-6-t_size_name[1]*2)
        img = cv2.putText(img, title, text_pos_title, 0, 0.7,  text_color, thickness=1, lineType=cv2.LINE_AA)



        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
    vis = Visdom()


    for i in range(len(imgs)):
        query_img = imgs[i]
        rank_imgs = []
        gallery_scale = len(res['score'][i])
        for j in [j0 for j0 in range(min(n_p, gallery_scale))] + [j0 for j0 in range(max(0, gallery_scale-n_n), gallery_scale)]:
            rank_img = cv2.imread(os.path.join(sample_path, res['person_name'][i][j], res['img_name'][i][j]))
            rank_img = cv2.cvtColor(rank_img, cv2.COLOR_BGR2RGB)
            rank_img = scale_resize(rank_img, show_img_size)

            # 在图片上添加name
            name = str(res['img_name'][i][j]).split('.')[0]
            text_color = (255,255,255)
            t_size_name = cv2.getTextSize(name, 0, fontScale=0.7, thickness=1)[0]
            text_pos_name = (int(rank_img.shape[1]/2)-int(t_size_name[0]/2), rank_img.shape[0]-2-t_size_name[1])
            rank_img = cv2.putText(rank_img, name, text_pos_name, 0, 0.7,  text_color, thickness=1, lineType=cv2.LINE_AA)


            # 在图片上添加相似度
            score = '%.4f'%res['score'][i][j]
            if j < n_p:
                text_color = (0,255,0)
            else:
                text_color = (255,0,0)
            t_size_score = cv2.getTextSize(score, 0, fontScale=0.7, thickness=1)[0]
            text_pos_score = (int(rank_img.shape[1]/2)-int(t_size_score[0]/2), rank_img.shape[0]-6-t_size_name[1]*2)
            rank_img = cv2.putText(rank_img, score, text_pos_score, 0, 0.7,  text_color, thickness=1, lineType=cv2.LINE_AA)





            rank_img = np.transpose(rank_img, (2, 0, 1))
            rank_imgs.append(rank_img)
        vis.images([query_img] + rank_imgs)

if __name__ == '__main__':
    import os, cv2

    test_imgs = []
    test_img_path = '/home/xzy/projects/People_In_Video/test_data/zhongtie_yanshou/people_query'
    sample_path =  '/home/xzy/projects/People_In_Video/test_data/zhongtie_yanshou/people_samples'

    # test_img_path = '/home/xzy/projects/People_In_Video/test_data/hurenVSyongshi/results_tracklets/clip0/tracklets/7'
    # sample_path =  '/home/xzy/projects/People_In_Video/test_data/hurenVSyongshi/people_samples'


    img_names = os.listdir(test_img_path)[32:64]
    for img_name in img_names:
        img = cv2.imread(os.path.join(test_img_path, img_name))
        test_imgs.append(img)

    demo(test_img_path, sample_path)