# -*- coding: utf-8 -*-
# python3
# Author: Haoyu Li, Zenghe Huang

import numpy as np
import pandas as pd
import os
from builtins import range, input
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
import pickle
from datetime import datetime
from sortedcontainers import SortedList
import wordcloud
from wordcloud import WordCloud, STOPWORDS

def main():
  isPreprocessing = 0
  if(isPreprocessing): # To get indexed movie id file
    preprocessing()
   # number of users and movies after we shrinking the dataset
  n = 1000 #users, keep it small to run fast
  m = 50 #movies, keep it small to run fast
  isGetPartialData = 0
  isDataVisulization = 0
  isUserBased = 1
  isItemBased = 0

  if(isGetPartialData):
    getPartialData(m,n)
  if(isDataVisulization):
    dataVisualization()
  
  # number of neighbors we want to take into consideration,(number of i')
  k = 25
  #If the two users have nothing in common, then we won't take into account, we need to set a threshold of minimum number of movies that two users have in common
  threshold = 5

  #Do the user-user based recommendation system
  if(isUserBased == 1):
    userBased(k,threshold) #O(n^2 m) time complexity, slow
  #Do the item-item based recommendation system
  if(isItemBased ==1):
    itemBased(k, threshold)
    # number of neighbors we want to take into consideration,(number of i')
  

#Data visualization is also used to generate the movie to title
def dataVisualization():
  ratingColumns = ['userId', 'movieId', 'rating','movieIndex']
  ratings = pd.read_csv('movielens-20m-dataset/partialRating.csv', sep=',', names=ratingColumns, usecols=range(4))
  movie_columns = ['movieId', 'title']
  movies = pd.read_csv('movielens-20m-dataset/movie.csv', sep=',', names=movie_columns, usecols=range(2), encoding="iso-8859-1")
  # create one merged DataFrame
  movieRatings = pd.merge(movies, ratings,on="movieId", how='inner')
  #Shwow the inner joined table
  print(movieRatings)
  movieToTitle = {}

  def updateByRows(row):
    title = row.title
    movie = row.movieIndex
    if movie not in movieToTitle:
      movieToTitle[movie] = [title] #If we movie id is not in the hash map, put into it
  
  movieRatings.apply(updateByRows,axis = 1) #Call the function updateByRows
  #Save as the binary files

  with open('movieToTitle.json','wb') as f:
    pickle.dump(movieToTitle,f)

  # Create a wordcloud of the movie titles
  
  movies['title'] = movieRatings['title'].fillna("").astype('str')
  title_corpus = ' '.join(set(movies['title']))
  
  wordCloud = WordCloud(stopwords=STOPWORDS, background_color='orange', height=1600, width=3000).generate(title_corpus)

  # Plot the wordcloud
  plt.figure(figsize=(10,8))
  plt.imshow(wordCloud)
  plt.axis('off')
  plt.show()
 

def preprocessing():
  data = pd.read_csv('movielens-20m-dataset/rating.csv')  #Load the original dataset
  data.userId = data.userId - 1 

  # The movie Id is not consecutive, therefore, we need to recreate the movie id file
  # We use a set here to eliminate duplicate
  uniqueMovieIds = set(data.movieId.values)
  indexedMovie = {}

  cnt = 0
  for ii in uniqueMovieIds:
    indexedMovie[ii] = cnt
    cnt += 1

   #Find data in original dataset and append to it
  data['movieIndex'] = data.apply(lambda row: indexedMovie[row.movieId], axis=1)

  #We do not need to use timestap right here, therefore we get rid of it
  data = data.drop(columns=['timestamp'])
  #save it to a csv file
  data.to_csv('movielens-20m-dataset/edited_rating.csv', index=False)

def getPartialData(m,n):
  exists = os.path.isfile('movielens-20m-dataset/partialRating.csv')
  if exists:
    # Store configuration file values

    # If the file has already existed, we will call the getDicts function to
    # get movieToUser Mapping, userToMovie Mapping and userMovieToRating mapping
    #Also in this function, we split the partial dataset into two parts, the training 
    #set and the testing set
    data =  pd.read_csv('movielens-20m-dataset/partialRating.csv')
    print("Generating the dictionaries right now...")
    getDicts(data)
  else:
    # Keep presets
    print("Partial Rating file does not exist, generating it right now...") 
  
    data = pd.read_csv('movielens-20m-dataset/edited_rating.csv')
    print("original dataframe size:", len(data))

    userCounts = Counter(data.userId)
    movieCounts = Counter(data.movieIndex)


    #We are trying to grab most common userids and movieids 
    #Since the most common function gives us a key-value pair, it gives us the id and the common number counts,
    #We only want to keep the ids
    userIds = [u for u, c in userCounts.most_common(n)]
    movieIds = [m for m, c in movieCounts.most_common(m)]

    # make a copy, otherwise it will just be pointers and won't be able to change 
    dataPartial = data[data.userId.isin(userIds) & data.movieIndex.isin(movieIds)].copy()

    # Since movie Ids and user Ids are not sequentials, we need to remark them
    userIdMap = {}
    idx = 0
    for ii in userIds:
      userIdMap[ii] = idx
      idx += 1

    movieIdMap = {}
    idx = 0
    for ii in movieIds:
      movieIdMap[ii] = idx
      idx += 1
     
    #Setting new Ids
    dataPartial.loc[:, 'userId'] = dataPartial.apply(lambda row: userIdMap[row.userId], axis=1)
    dataPartial.loc[:, 'movieIndex'] = dataPartial.apply(lambda row: movieIdMap[row.movieIndex], axis=1)
 
    print("max userId:", dataPartial.userId.max())
    print("max movieId:", dataPartial.movieIndex.max())

    dataPartial.to_csv('movielens-20m-dataset/partialRating.csv', index=False)


def getDicts(data):
  n = data.userId.max()+1
  m = data.movieIndex.max()+1

  #We need to split the data into the training set and testing set, therefore we need to shuffle the data at first
  data = shuffle(data)
  #We want to use 80% of our data as the training set, and 20% as the testing set
  training = data.iloc[:int(0.8*len(data))]
  testing = data.iloc[int(0.8*len(data)):]

  #We need a dictionary to tell the user to movie mappings
  userToMovie = {}
  movieToUser = {} #Movie to User Mapping
  userMovieToRating = {} # Get user's rating to a specific movie
  #For each row, call the function updateByRows
  userMovieToRatingTest = {} # For testing set, get rating of a user on specific movie

  def updateByRows(row):
    user = int(row.userId)
    movie = int(row.movieIndex)
    if user not in userToMovie:
      userToMovie[user] = [movie] #If we haven't met user ii yet, we create a new mapping, and put the corresponding movie jj that user ii has rated into the mapping
    else:
      userToMovie[user].append(movie) #If user ii has already existed, we append movie jj to user ii

    #We do the similar thing for movie to user
    if movie not in movieToUser:
      movieToUser[movie] = [user]
    else:
      movieToUser[movie].append(user)

    userMovieToRating[(user,movie)] = row.rating

  #Function for testing set, get rating for user-movie pair
  def updateByRowsTest(row):
    user = int(row.userId)
    movie = int(row.movieIndex)
    userMovieToRatingTest[(user,movie)] = row.rating

  training.apply(updateByRows,axis = 1) #Call the function for training test
  testing.apply(updateByRowsTest,axis = 1) #Call the function for testing set

  #Save as the binary files
  with open('userToMovie.json','wb') as f:
    pickle.dump(userToMovie,f)
  
  with open('movieToUser.json','wb') as f:
    pickle.dump(movieToUser,f)

  with open('userMovieToRating.json','wb') as f:
    pickle.dump(userMovieToRating,f)
  
  with open('userMovieToRatingTest.json','wb') as f:
    pickle.dump(userMovieToRatingTest,f)
  
  print("Complete")
  print("Now we have:\n userToMovie.json \n movieToUser.json \n userMovieToRating.json \n userMovieToRatingTest.json ")


############################################################################################################
#USER USER RECOMENDATION SYSTEM STARTS HERE
def userBased(k, threshold):
  if not os.path.exists('userToMovie.json') or not os.path.exists('movieToUser.json') or not os.path.exists('userMovieToRating.json') or not os.path.exists('userMovieToRatingTest.json'):
    #If either of the above files do not exist, we need to call the getDics function to generate these files
    data =  pd.read_csv('movielens-20m-dataset/partialRating.csv')
    getDicts(data)
  else:
    with open('userToMovie.json','rb') as f:
      userToMovie = pickle.load(f)
    with open('movieToUser.json','rb') as f:
      movieToUser = pickle.load(f)
    with open('userMovieToRating.json','rb') as f:
      userMovieToRating = pickle.load(f)
    with open('userMovieToRatingTest.json','rb') as f:
      userMovieToRatingTest = pickle.load(f)
    with open('movieToTitle.json','rb') as f:
      movieToTitle = pickle.load(f)
    
    #get number of users and movies, since we splitted the shrinked dataset into the training set and the testing set, we need 
    #calculate the max key between the two set for both user and movie
    #number of users
    print('User User Based Approach')
    n1 = np.max(list(userToMovie.keys()))
    n2 = np.max([user for (user,movie), r in userMovieToRatingTest.items()])
    m1 = np.max(list(movieToUser.keys()))
    m2 = np.max([movie for (user,movie), r in userMovieToRatingTest.items()])
    n = max(n1,n2)+1
    m = max(m1,m2)+1
    print("number of users is", n)
    print("number of movies is", m)
    
    neighbors = [] 
    avg = [] # Avg rating for each user
    dev = [] # dev for each user

    for user in range(n):
      #Find movie that user i watched 
      movieI = set(userToMovie[user]) #use set  to eliminate duplicate
      ratingI = { movie:userMovieToRating[(user,movie)] for movie in movieI }
      #average rating that user i gave on movies that he/she watched
      avgI = np.mean(list(ratingI.values()))
      #deviation that user i on each movie
      devI = {movie:(rating - avgI) for movie,rating in ratingI.items() }
      devIValues = np.array(list(devI.values()))
      sigmaI = np.sqrt(devIValues.dot(devIValues))
      avg.append(avgI) # For all users
      dev.append(devI) # For all users
      #For each user, we want to calculate the user-user weight, so for each weight we calculated, we put that into the sorted list, and by the end of this calculation, we just want to keep the top K entries
      sl = SortedList()
      for userJ in range(n):
        if userJ!=user: #We do not want to include user itself in the calculation
          movieJ = set(userToMovie[userJ])
          #Movie I and Movie J are two sets of movie that the two users watched, use & will give us the intersection between two sets
          commonSeen = (movieI & movieJ)
          if(len(commonSeen)> threshold): #We only take into consideration if two users have commonly seen movie greater than the threshold
            ratingJ = {movie:userMovieToRating[(userJ,movie)] for movie in movieJ}
            avgJ = np.mean(list(ratingJ.values()))
            devJ = {movie:(rating-avgJ) for movie,rating in ratingJ.items() } #Movie-dev pairs
            devJValues = np.array(list(devJ.values()))  #Only get the deviations
            sigmaJ = np.sqrt(devJValues.dot(devJValues))
            
            #Calculate the correlation coefficient using pearson similarity
            weight = sum(devI[movie]*devJ[movie] for movie in commonSeen)/(sigmaI*sigmaJ)
            #The higher the weight, the better
            #Sorted list is in ascending order, therefore, in order to make the one with highest weight in the front, we need to make the weights negative
            sl.add((-weight,userJ))
            if(len(sl)> k): #We only want to keep top k users
              del sl[-1]
      neighbors.append(sl)  #Store the neighbors 
      #print(user)

  def prediction(userI,movie):
    numerator = 0
    denominator = 0
    #The weight is initially negative, therfore, we need to negative it back
    for negWeight,userJ in neighbors[userI]:
      try:
        numerator += (-negWeight)*dev[userJ][movie]
        denominator += abs(negWeight)
      except KeyError:
        #User j might not have watched the same movie, therefore, when this case happens, we just pass
        pass
    if denominator == 0:
      prediction = avg[userI] 
    else:
      prediction = numerator/denominator + avg[userI]
      #Notice that prediction can only inbetween 0.5-5, therefore, anything less than that will be scaled to 0.5,and anything greater than that will be scaled to 5
    prediction = min(5,prediction)
    prediction = max(0.5,prediction)
    return prediction


  trainPrediction = []
  trainTarget = []
  testPrediction = []
  testTarget = []
  for (userI,movie), target in userMovieToRating.items():
    pred = prediction(userI,movie)
    trainPrediction.append(pred)
    trainTarget.append(target)

  for(userI,movie),target in userMovieToRatingTest.items():
    pred = prediction(userI,movie)
    testPrediction.append(pred)
    testTarget.append(target)

  print('train mean square error:',mse(trainPrediction,trainTarget))
  print('test mean square error:',mse(testPrediction,testTarget))
  print('==========================================')
  print('Some test Case:')

  print('Prediction for user 0 based on other users')
  print('Actual')
  movie0 = set(userToMovie[0])
  for movie in movie0:
    print('movie:',movieToTitle[str(movie)],":",userMovieToRating[(0,movie)])
  print('Predicted:')
  for movie in range(m):
    print('movie:',movieToTitle[str(movie)],':',round(prediction(0,movie),2))
  print('Done!!!')
  


#function used to calculate the mean square error
def mse(prediction,target):
  prediction = np.array(prediction)
  target = np.array(target)
  return np.mean((prediction-target)**2)


############################################################################################################
#ITEM ITEM BASED RECOMENDATION SYSTEM STARTS HERE

#This is basically the same as the user-user collabrative filter
def itemBased(k,threshold):
  if not os.path.exists('userToMovie.json') or not os.path.exists('movieToUser.json') or not os.path.exists('userMovieToRating.json') or not os.path.exists('userMovieToRatingTest.json'):
    #If either of the above files do not exist, we need to call the getDics function to generate these files
    data =  pd.read_csv('movielens-20m-dataset/partialRating.csv')
    getDicts(data)
  else:
    with open('userToMovie.json','rb') as f:
      userToMovie = pickle.load(f)
    with open('movieToUser.json','rb') as f:
      movieToUser = pickle.load(f)
    with open('userMovieToRating.json','rb') as f:
      userMovieToRating = pickle.load(f)
    with open('userMovieToRatingTest.json','rb') as f:
      userMovieToRatingTest = pickle.load(f)
    with open('movieToTitle.json','rb') as f:
      movieToTitle = pickle.load(f)
    print('Item Item based Approach')
    n1 = np.max(list(userToMovie.keys()))
    n2 = np.max([user for (user,movie), r in userMovieToRatingTest.items()])
    m1 = np.max(list(movieToUser.keys()))
    m2 = np.max([movie for (user,movie), r in userMovieToRatingTest.items()])
    n = max(n1,n2)+1
    m = max(m1,m2)+1
    print("number of users is", n)
    print("number of movies is", m)

    neighbors = [] 
    avg = [] # Avg rating for each user
    dev = [] # dev for each user

    for movie in range(m):
      userI = set(movieToUser[movie]) #get all users who watched movie i before
      ratingI = {user:userMovieToRating[(user,movie)] for user in userI} #Get ratings that they have rated on that movie
      avgI = np.mean(list(ratingI.values())) #Calculate the average that the users have gave on that movie
      #Calculate deviations that users gave on that movie
      devI = {user:(rating-avgI) for user, rating in ratingI.items() }
      devIValues = np.array(list(devI.values()))
      sigmaI = np.sqrt(devIValues.dot(devIValues))

      #Save average for all movies into an array
      avg.append(avgI)
      dev.append(devI)

      #get an sortedlist object
      sl = SortedList()
      for movieJ in range(m):
        if movieJ!=movie:
          userJ = set(movieToUser[movieJ])
          commonUser = (userI & userJ)
          if( len(commonUser) > threshold):
            ratingJ = {user:userMovieToRating[(user,movieJ)] for user in userJ}
            avgJ = np.mean(list(ratingJ.values()))
            devJ = {user:(rating-avgJ) for user,rating in ratingJ.items() }
            devJValues = np.array(list(devJ.values()))
            sigmaJ = np.sqrt(devJValues.dot(devJValues))

            #Calculate the correlation coefficient
            weight = sum(devI[m]*devJ[m] for m in commonUser)/(sigmaI*sigmaJ)

            sl.add((-weight,movieJ))
            if( len(sl) > k):
              del sl[-1]
            
      neighbors.append(sl)
      #print which user we are processing right now
      print("movie",movie)

    def prediction(movieI,user):
      numerator = 0
      denominator = 0
      for negWeight, movie in neighbors[movieI]:
        try:
         # print(movie)
         # print(user)
          numerator += (-negWeight) * dev[movie][user]
          denominator += abs(negWeight)
        except KeyError:
          pass
      
      if(denominator == 0):
        prediction = avg[movieI]
      else:
        prediction = numerator/denominator + avg[movieI]
      prediction = min(prediction,5)
      prediction = max(prediction,0.5)
      return prediction
    
    trainPrediction = []
    trainTarget = []
    testPrediction = []
    testTarget = []
    for (user,movie), target in userMovieToRating.items():
      pred = prediction(movie,user)
      trainPrediction.append(pred)
      trainTarget.append(target)

    for(user,movie),target in userMovieToRatingTest.items():
      pred = prediction(movie,user)
      testPrediction.append(pred)
      testTarget.append(target)
    
    print('train mean square error:',mse(trainPrediction,trainTarget))
    print('test mean square error:',mse(testPrediction,testTarget))
    print('==========================================')
    print('Some test Cases:')

    print('Prediction for user 0 using item-item approach')
    print('Actual')
    movie0 = set(userToMovie[0])

    actualMovie0 = []
    actualRating0 = []
    for movie in movie0:
      actualMovie0.append(movie)
      actualRating0.append(userMovieToRating[(0,movie)])
      print('movie:',movieToTitle[str(movie)],":",userMovieToRating[(0,movie)])
    print('Predicted:')
    
    title0 = []
    rating0 = []

    for movie in range(m):
      title0.append(movieToTitle[str(movie)])
      rating0.append(prediction(movie,0))
      print('movie:',movieToTitle[str(movie)],':',round(prediction(movie,0),2))
    print('Done!!!')

    #Plot the histogram

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(30,12))

    y_pos = np.arange(len(title0))
    ax.barh(y_pos, rating0, align='center',
            color = '#66c2ff',label = 'predicted',alpha = 0.8)
    # Create twin axes
    axb = ax.twiny()
    axb.barh(actualMovie0, actualRating0, align='center',
            color = '#ffcc00',label = 'user-rated',alpha = 0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(title0)
    ax.invert_yaxis()  
    ax.set_xlabel('Predicted Ratings  vs User Ratings')
    plt.axis('off')
    plt.show()



      



if __name__== "__main__":
  main()