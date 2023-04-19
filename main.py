from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse,HTMLResponse
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.templating import Jinja2Templates
import torch
import pandas as pd
import cv2 as cv
import open3d as o3d
import numpy as np

app = FastAPI()
model = torch.hub.load('yolov5', 'custom', path='bestmixed.pt', source='local')
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#Create a folder to store uploaded files
if not os.path.exists("uploads"):
    os.mkdir("uploads")

# Endpoint to upload a file
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    dir = os.listdir("uploads")
    # print(dir)
    if os.path.exists("uploads"):
        for f in os.listdir("uploads"):
            os.remove(os.path.join("uploads", f))

    # Save the uploaded file to the 'uploads' folder
    with open(f"uploads/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())
    
    list_min = []
    file_name=f"uploads/{file.filename}"
    cloud = o3d.io.read_point_cloud(file_name)
    S = -10*(np.pi/180)
    R = cloud.get_rotation_matrix_from_xyz((0, S, 0))
    cloud = cloud.rotate(R, center=(0,0,0))
    array=np.array(cloud.points)
    # print(array)
    array1=array*800
    a=array1.astype(dtype=int)
    df2=pd.DataFrame(a,columns=['x','y','z'])
    df3=df2.drop(['x'],axis=1)
    # print(df3['y'].min())
    list_min.append(df3['y'].min())
    df3['y']=df3['y']-df3['y'].min()
    df3['z']=df3['z']-df3['z'].min()
    df4=df3
    blank_image = 255*np.ones((800,800,3), dtype = np.uint8)
    df4.to_csv("y_z plane data.csv")
    #image writing on blank image
    for ind in df4.index:
        image = cv.circle(blank_image, (((df4['z'][ind])),800-(df4['y'][ind])), radius=1, color=(255, 0, 0), thickness=-1)
    # print("blank image")

    cv.imwrite(file_name+'.jpg', image)

    x = (file_name+'.jpg')
    print(x)
    try:
        results = model(x)
        
    #calculation of under bust height from bounding box
        y_max = (results.pandas().xyxy[0].loc[0,'ymax'])
        # print(y_max)
    #underbust value calculation 
        ub_height=(800+list_min[0]-(results.pandas().xyxy[0].loc[0,'ymax']))/800
        # print("underbust detected using ML",ub_height)
    except:
        y_max=0
        ub_height=0
        # print("UB not detected ")

    for ind in df4.index:
        image1 = cv.circle(image, ((df4['z'][ind]),(int(y_max))), radius=1, color=(0, 255, 0), thickness=-1) #ml model
        
    cv.imwrite(file_name+'m'+'.jpg', image1)
    global fname2
    fname2=(file_name+'m'+'.jpg')
      
    print("ub_height",ub_height)

    return file_name[7:],ub_height


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8011)



