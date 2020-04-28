import shutil
import os
from imgprep import ImagePreprocessing
import pandas as pd
 		
class ImageDataPreparation(object):
	def __init__(self, imgPrep, imgAugm, 
			  dataFramePath, IdCol, CatCol, 
			  SrcDir,
			  distTestOrigDir, distTestAugmDir, 
			  distTrainOrigDir, distTrainAugmDir):
		self.ConfigurationValid = True
		self.Table = None
		if not os.path.isfile(dataFramePath):
			self.ConfigurationValid = False
		else:
			self.Table = pd.read_csv(dataFramePath)
			for i in range(256):
				self.Table = self.Table.sample(frac = 1).reset_index(drop=True)
		self.ImgPrep = imgPrep
		self.ImgAugm = imgAugm
		self.Id = IdCol
		self.Cat = CatCol
		self.SrcDir = SrcDir
		self.DistTestOrigDir = distTestOrigDir
		self.DistTestAugmDir = distTestAugmDir
		self.DistTrainOrigDir = distTrainOrigDir
		self.DistTrainAugmDir = distTrainAugmDir
		self.ConfigurationValid = self.ValidateArguments()

	def ValidateArguments(self):
		if self.ImgPrep is None or self.ImgAugm is None:
			return False
		if not ((self.Id in self.Table.columns) and (self.Cat in self.Table.columns)):
			return False
		if not os.path.exists(self.SrcDir):
			return False
		if not self.DistTestOrigDir or not self.DistTestAugmDir or not self.DistTrainOrigDir or not self.DistTrainAugmDir:
			return False
		os.makedirs(self.DistTestOrigDir,exist_ok = True)
		os.makedirs(self.DistTestAugmDir,exist_ok = True)
		os.makedirs(self.DistTrainOrigDir,exist_ok = True)
		os.makedirs(self.DistTrainAugmDir,exist_ok = True)
		return True    
		
	def __call__(self,split = 0.1, trCnt = 0, tsCnt = 0, rewrite = False):
		if not self.ConfigurationValid:
			return False
		catv = self.Table[self.Cat].unique()
		catv.sort()
		for ik in catv:
			trainv, testv = self._get_single_train_test(ik,split)			
			path = os.path.join(self.DistTrainOrigDir,str(ik))
			__class__._ensure_subdirectory(path,rewrite)
			self._transform_images(trainv,ik,path,False)

			path = os.path.join(self.DistTestOrigDir,str(ik))
			__class__._ensure_subdirectory(path,rewrite)
			self._transform_images(testv,ik,path,False)
			left = trCnt - len(trainv)
			if left > 0:
				trainv = trainv * (1 + left // len(trainv))
				trainv = trainv[:left]
				path = os.path.join(self.DistTrainAugmDir,str(ik))
				__class__._ensure_subdirectory(path,rewrite)
				self._transform_images(trainv,ik,path,True)

			left = tsCnt - len(testv)
			if left > 0:
				testv = testv * (1 + left // len(testv))
				testv = testv[:left]
				path = os.path.join(self.DistTestAugmDir,str(ik))
				__class__._ensure_subdirectory(path,rewrite)
				self._transform_images(testv,ik,path,True) 	
		return True
			
	def _ensure_subdirectory(path, rewrite = False):
		if rewrite and os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path,exist_ok = True)
		
	def _transform_images(self,filev, cat, distDir, augmented = False):
		ik = 1
		for file in filev:
			if not os.path.exists(file):
				continue
			_, fileName = os.path.split(file)
			if augmented:
				fileName = 'aug_{}_{}.png'.format(cat,ik)
				ik += 1
			dest = os.path.join(distDir,fileName)
			img = ImagePreprocessing.LoadImage(file)
			if augmented:
				img = self.ImgAugm(img)
			self.ImgPrep(img)
			self.ImgPrep.SaveModifiedImage(dest)
					
	def _get_single_train_test(self, cat, split = 0.1):
		df = self.Table[self.Table[self.Cat] == cat].copy().reset_index(drop = True)
		trainsz = df.shape[0]
		testsz = 1 + int(split * trainsz)
		trainsz -= testsz
		trainv = [os.path.join(self.SrcDir,x + '.png') for x in df.loc[:trainsz-1 ,'id_code']]
		testv = [os.path.join(self.SrcDir,x + '.png') for x in df.loc[trainsz:,'id_code']]
		return (trainv, testv)

class DataFramePreparation(object):
	def __init__(self, trCnt, trDfFilePath, destTrainOrigDir, destTrainAugmDir,
			 tsCnt,tsDfFilePath, destTestOrigDir, destTestAugmDir):
		self.trCnt = trCnt
		self.trDfFilePath = trDfFilePath
		self.destTrainOrigDir = destTrainOrigDir
		self.destTrainAugmDir = destTrainAugmDir
		self.tsCnt = tsCnt
		self.tsDfFilePath = tsDfFilePath
		self.destTestOrigDir = destTestOrigDir
		self.destTestAugmDir = destTestAugmDir

	def __call__(self, catv):
		if catv is None:
			return False
		if os.path.isfile(self.trDfFilePath):
			os.remove(self.trDfFilePath)
		train_df = __class__._make_new_table(self.trCnt,catv, self.destTrainOrigDir,self.destTrainAugmDir)
		train_df.to_csv(self.trDfFilePath)

		if os.path.isfile(self.tsDfFilePath):
			os.remove(self.tsDfFilePath)
		test_df = __class__._make_new_table(self.tsCnt,catv, self.destTestOrigDir,self.destTestAugmDir)
		test_df.to_csv(self.tsDfFilePath)
		return True

	def _make_new_table(maxCnt, catv, origDir, augmDir):
		rootDir = os.path.commonpath([origDir, augmDir])
		dictpd = {'filename' : [], 'class' : [], 'type' : []}
		if maxCnt == 0:
			return pd.DataFrame(dictpd)
		relorigDir = os.path.relpath(origDir, rootDir)
		relaugmDir = os.path.relpath(augmDir, rootDir)
		for i in catv:
			trgorigdir = os.path.join(origDir, str(i))
			trgaugmdir = os.path.join(augmDir, str(i))
			origFilenames = [os.path.join(relorigDir,'./{}/{}'.format(i,f)) for f in os.listdir(trgorigdir) if os.path.isfile(os.path.join(trgorigdir,f))]
			types = [0] * len(origFilenames)
			augFilenames = [os.path.join(relaugmDir,'./{}/{}'.format(i,f)) for f in os.listdir(trgaugmdir) if os.path.isfile(os.path.join(trgaugmdir,f))]
			augFilenames.sort()
			extendLen = maxCnt - len(origFilenames)
			if extendLen > len(augFilenames):
				return None
			if extendLen <= 0:
				origFilenames = origFilenames[:maxCnt]
				types = [0] * maxCnt
			for j in range(extendLen):
				origFilenames.append(augFilenames[j])
				_, filename = os.path.split(augFilenames[j])
				filename = os.path.splitext(filename)[0]
				types.append(int(filename.split('_')[2]))
			dictpd['filename'] += origFilenames
			dictpd['class'] += [i] * len(types) 
			dictpd['type'] += types
		return pd.DataFrame(dictpd)