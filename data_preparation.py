import glob, imageio, shutil, nibabel
from pathlib import Path

#selecting 20 slices with max file size
def select_slice(image_array):

    imageSizeList = []
    totalSlices = image_array.shape[2]
    tempPath = Path.home() / "temp"
    Path(tempPath).mkdir(exist_ok = True)

    for slice in range(0, totalSlices) :
        data = image_array[:, :, slice]
        imagePath = str(tempPath) + "/" + str(slice+1) + ".png"
        imageio.imwrite(imagePath, data)
        imageSizeList.append(Path(imagePath).stat().st_size)

    shutil.rmtree(tempPath)
    maxSizeList = []

    for i in range(0, 20) :
        maxSizeList.append(imageSizeList.index(max(imageSizeList))+1)
        imageSizeList.remove(max(imageSizeList))

    return maxSizeList


def niitopng(inputFilePath, outputFolderPath, counter, mode, sliceList):

    if mode == 1:
        for slice in sliceList :
            image_array = nibabel.load(inputFilePath).get_data()
            data = image_array[:, :, slice-1]

            if counter+1 < 10 :
                image_name = str(outputFolderPath) + "/frame_0000" + str(counter+1) + ".png"
            elif counter+1 < 100 :
                image_name = str(outputFolderPath) + "/frame_000" + str(counter+1) + ".png"
            elif counter+1 < 1000:
                image_name = str(outputFolderPath) + "/frame_00" + str(counter+1) + ".png"
            elif counter+1 < 10000:
                image_name = str(outputFolderPath) + "/frame_0" + str(counter+1) + ".png"
            else :
                image_name = str(outputFolderPath) + "/frame_" + str(counter+1) + ".png"

            imageio.imwrite(image_name, data)
            counter += 1

    if mode == 2 :
        for slice in sliceList :
            image_array = nibabel.load(inputFilePath).get_data()
            data = image_array[:, :, slice-1]

            if counter+1 < 10 :
                image_name = str(outputFolderPath) + "/mask_0000" + str(counter+1) + ".png"
            elif counter+1 < 100 :
                image_name = str(outputFolderPath) + "/mask_000" + str(counter+1) + ".png"
            elif counter+1 < 1000:
                image_name = str(outputFolderPath) + "/mask_00" + str(counter+1) + ".png"
            elif counter+1 < 10000:
                image_name = str(outputFolderPath) + "/mask_0" + str(counter+1) + ".png"
            else :
                image_name = str(outputFolderPath) + "/mask_" + str(counter+1) + ".png"

            imageio.imwrite(image_name, data.astype('uint8'))
            counter += 1

    if mode == 3 :
        for slice in sliceList :
            image_array = nibabel.load(inputFilePath).get_data()
            data = image_array[:, :, slice-1]

            if counter+1 < 10 :
                image_name = str(outputFolderPath) + "/val_frame_0000" + str(counter+1) + ".png"
            elif counter+1 < 100 :
                image_name = str(outputFolderPath) + "/val_frame_000" + str(counter+1) + ".png"
            elif counter+1 < 1000:
                image_name = str(outputFolderPath) + "/val_frame_00" + str(counter+1) + ".png"
            elif counter+1 < 10000:
                image_name = str(outputFolderPath) + "/val_frame_0" + str(counter+1) + ".png"
            else :
                image_name = str(outputFolderPath) + "/val_frame_" + str(counter+1) + ".png"

            imageio.imwrite(image_name, data)
            counter += 1

    if mode == 4 :
        for slice in sliceList :
            image_array = nibabel.load(inputFilePath).get_data()
            data = image_array[:, :, slice-1]

            if counter+1 < 10 :
                image_name = str(outputFolderPath) + "/val_mask_0000" + str(counter+1) + ".png"
            elif counter+1 < 100 :
                image_name = str(outputFolderPath) + "/val_mask_000" + str(counter+1) + ".png"
            elif counter+1 < 1000:
                image_name = str(outputFolderPath) + "/val_mask_00" + str(counter+1) + ".png"
            elif counter+1 < 10000:
                image_name = str(outputFolderPath) + "/val_mask_0" + str(counter+1) + ".png"
            else :
                image_name = str(outputFolderPath) + "/val_mask_" + str(counter+1) + ".png"

            imageio.imwrite(image_name, data.astype('uint8'))
            counter += 1

    if mode == 5 :
        for slice in sliceList :
            image_array = nibabel.load(inputFilePath).get_data()
            data = image_array[:, :, slice-1]

            if counter+1 < 10 :
                image_name = str(outputFolderPath) + "/test_frame_0000" + str(counter+1) + ".png"
            elif counter+1 < 100 :
                image_name = str(outputFolderPath) + "/test_frame_000" + str(counter+1) + ".png"
            elif counter+1 < 1000:
                image_name = str(outputFolderPath) + "/test_frame_00" + str(counter+1) + ".png"
            elif counter+1 < 10000:
                image_name = str(outputFolderPath) + "/test_frame_0" + str(counter+1) + ".png"
            else :
                image_name = str(outputFolderPath) + "/test_frame_" + str(counter+1) + ".png"

            imageio.imwrite(image_name, data)
            counter += 1

    if mode == 6 :
        for slice in sliceList :
            image_array = nibabel.load(inputFilePath).get_data()
            data = image_array[:, :, slice-1]

            if counter+1 < 10 :
                image_name = str(outputFolderPath) + "/test_mask_0000" + str(counter+1) + ".png"
            elif counter+1 < 100 :
                image_name = str(outputFolderPath) + "/test_mask_000" + str(counter+1) + ".png"
            elif counter+1 < 1000:
                image_name = str(outputFolderPath) + "/test_mask_00" + str(counter+1) + ".png"
            elif counter+1 < 10000:
                image_name = str(outputFolderPath) + "/test_mask_0" + str(counter+1) + ".png"
            else :
                image_name = str(outputFolderPath) + "/test_mask_" + str(counter+1) + ".png"

            imageio.imwrite(image_name, data.astype('uint8'))
            counter += 1



if __name__ == "__main__":
    datasetPath = input("Enter Path of the Dataset: ")

    outputTrainPath = Path(datasetPath.replace("brats19","Dataset/train_frames/train"))
    Path(outputTrainPath).mkdir(parents = True, exist_ok = True)
    outputTrainSegPath = Path(datasetPath.replace("brats19","Dataset/train_masks/train"))
    Path(outputTrainSegPath).mkdir(parents = True, exist_ok = True)

    outputValPath = Path(datasetPath.replace("brats19","Dataset/val_frames/val"))
    Path(outputValPath).mkdir(parents = True, exist_ok = True)
    outputValSegPath = Path(datasetPath.replace("brats19","Dataset/val_masks/val"))
    Path(outputValSegPath).mkdir(parents = True, exist_ok = True)

    outputTestPath = Path(datasetPath.replace("brats19","Dataset/test_frames/test"))
    Path(outputTestPath).mkdir(parents = True, exist_ok = True)
    outputTestSegPath = Path(datasetPath.replace("brats19","Dataset/test_masks/test"))
    Path(outputTestSegPath).mkdir(parents = True, exist_ok = True)

    allT1Brains = Path(datasetPath).rglob("*t1.nii.gz")
    allT1ceBrains = Path(datasetPath).rglob("*t1ce.nii.gz")
    allT2Brains = Path(datasetPath).rglob("*t2.nii.gz")
    allFlairBrains = Path(datasetPath).rglob("*flair.nii.gz")

    allBrainsStr = []
    for brain in allT1Brains :
        allBrainsStr.append(str(brain))
    for brain in allT1ceBrains :
        allBrainsStr.append(str(brain))
    for brain in allT2Brains :
        allBrainsStr.append(str(brain))
    for brain in allFlairBrains :
        allBrainsStr.append(str(brain))

    #dataset split into 70-20-10 ratio (train-val-test)
    trainSplit = int(0.7*(len(allBrainsStr)))
    valSplit = int(0.9*(len(allBrainsStr)))

    allTrainBrains = allBrainsStr[:trainSplit]
    allValBrains = allBrainsStr[trainSplit:valSplit]
    allTestBrains = allBrainsStr[valSplit:]

    allSegBrains = Path(datasetPath).rglob("*seg.nii.gz")
    allSegBrainsStr1 = []
    for brain in allSegBrains :
        allSegBrainsStr1.append(str(brain))

    allSegBrainsStr2 = allSegBrainsStr1
    allSegBrainsStr3 = allSegBrainsStr1
    allSegBrainsStr4 = allSegBrainsStr1

    allSegBrainsStr = allSegBrainsStr1 + allSegBrainsStr2 + allSegBrainsStr3 + allSegBrainsStr4


    allTrainSegBrains =  allSegBrainsStr[:trainSplit]
    allValSegBrains = allSegBrainsStr[trainSplit:valSplit]
    allTestSegBrains = allSegBrainsStr[valSplit:]

    SliceIndexList = []
    maxSliceList = []
    for brain in allSegBrainsStr :
        image_array = nibabel.load(brain).get_data()
        maxSliceList = select_slice(image_array)
        SliceIndexList.append(maxSliceList)

    counter = 0
    for index, brain in enumerate(allTrainBrains):
        niitopng(brain, outputTrainPath, counter, 1, SliceIndexList[index])
        counter += 20

    counter = 0
    for index, brain in enumerate(allTrainSegBrains):
        niitopng(brain, outputTrainSegPath, counter, 2, SliceIndexList[index])
        counter += 20

    counter = 0
    for index, brain in enumerate(allValBrains):
        niitopng(brain, outputValPath, counter, 3, SliceIndexList[index + trainSplit])
        counter += 20

    counter = 0
    for index, brain in enumerate(allValSegBrains):
        niitopng(brain, outputValSegPath, counter, 4, SliceIndexList[index + trainSplit])
        counter += 20

    counter = 0
    for index, brain in enumerate(allTestBrains):
        niitopng(brain, outputTestPath, counter, 5, SliceIndexList[index + valSplit])
        counter += 20

    counter = 0
    for index, brain in enumerate(allTestSegBrains):
        niitopng(brain, outputTestSegPath, counter, 6, SliceIndexList[index + valSplit])
        counter += 20

    print("finished converting images")
