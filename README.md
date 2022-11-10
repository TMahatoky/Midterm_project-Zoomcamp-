# Midterm_project-Zoomcamp-
Fake job posting classification 

The dataset can be retrieved on : https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs

In the world of job posting, it is pretty clear that for those that are desperately hunting for jobs all the offers available
are worth getting into. However, among all those job offers, there are frauds that would bring no benefits at all and in fact would actually waste precious time. This dataset provides 18K job descriptions out of which about 800 are fake, and using the location feature as well as whether the offer has the possibility of telecommuting, if the company has a logo, if there is any question as well as taking into account the employment type, we wish to build a model that would allow us to predict whether the fraudulent column is true or not. The model will thus allow us to verify based on these features, if we should trust or avoid this offer. 

The predict-test.py file contains a sample offer to test the model. 

dockerbuilding: 
  - docker build -t midproject .
  - docker run -it --rm -p 9696:9696 midproject

To run the project after dockerbuilding: 
  - First option : * pipenv install waitress (is needed if on windows)   
                   * pipenv run waitress-serve --listen=:0.0.0.0:9696 predict:app
  
  - Second option: * pipenv install gunicorn (is needed if on linux)
                   * pipenv run gunicorn --bind 0.0.0.0:9696 predict:app 
   
  
