# ISM Project Group 2

## Setup 

- clone the git repository 
- From [this webisde](https://cloud.tuhh.de/index.php/s/76AsGdY34NkQQ5c?dir=/Project/Baselines) 
    - install the images missing in :
        - phase_1a/images/
        - phase_2a/images/
        - phase_1b/images/
        - phase_2b/images/
    - install the missing file: phase_2b/submission/model/model.safetensors  

## Using git while avoiding merge conflicts

### 1. Update local main
git checkout main
git pull origin main

### 2. Create feature branch
git checkout -b feature-name

### 3. Work locally
git add .
git commit -m "Clear message"

### 4. Push branch
git push origin feature-name

### 5. Create PR on GitHub → assign reviewer → merge after approval

### 6. Cleanup
git checkout main
git pull origin main
git branch -d feature-name
git push origin --delete feature-name