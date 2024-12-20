Note- scraping and cleaning data is housed in our private repo (data here is static- last updated as of 12/18/2024); please contact us with any questions/ if you want access to the src code. Thanks!


## SpellScribe 🪄
#### Welcome witches, wizards (& educated muggles)! We've received several messages here at the Daily Prophet asking, no begging, to provide an agent who can answer all your magical queries. We are here to say that your calls have been heard! We present to you: SpellScribe! Want to get back at your foe? SpellScribe! Lost your waterbottle and can't remember where you last had it, and now you are facing a 3-headed dog? Oddly specific, but SpellScribe has a solution for you! 
Please use responsibly! 🪄 

### Software requirements:

Python,
NodeJS,
pnpm,
SQL,
MySQL Workbench (easier to upload csv)


## To run:

Frontend: https://final-project-git-main-dravind2s-projects.vercel.app


Backend:
Because we use an openai key & private database at Hopkins, we have made the backend locally runnable and not deployable. Please upload potions.csv (backend/scraper/data/potions.csv) and spells.csv (backend/scraper/data/spells.csv) to your SQL database. 

#### Make a copy of the .env_copy file and change it to .env

Fill in the environment variables in the file.


#### After doing so, run the following commands in your terminal

```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Now, you have successfully installed the packages necessary to run the backend.

#### To launch your server, please run:


```
cd backend
python main.py
```

Congratulations! You should get the following message:

```
INFO:     Started server process [73052]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

You can now launch the frontend and start discovering spells and potions alike to fulfill your desires! 
