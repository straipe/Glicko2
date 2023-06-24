# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import math
import random
import sqlite3
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
# %matplotlib inline

# +
conn = sqlite3.connect(':memory:') # memory DB 설정
c = conn.cursor()

#사용자 정보 DB 생성
c.execute('''CREATE TABLE users
             (id integer primary key, rating integer, skill integer, rd real, win integer, lose integer, draw integer, sigma real)''')

#매칭 큐 DB 생성
c.execute('''CREATE TABLE queue
             (id integer primary key, rating integer, skill integer, rd real, win integer, lose integer, draw integer, sigma real)''')

# -

# <h1>Generate Users</h1>

#사용자 전적 레코드 DB 생성
for i in range(1,10001):
    c.execute('''CREATE TABLE user'''+str(i)+'''_record
                 (record_num integer primary key autoincrement, id integer, opp_id integer, opp_rating integer, opp_rd real, result real)''')
c.execute("select name from sqlite_master where type='table'")
table_list=c.fetchall()
table_list[-10:]

mu = 1500
sigma = 400
size = 10000
skills = np.random.normal(mu, sigma, size) #평균 1500, 표준편차 400인 정규분포 내 난수 10000개가 담긴 배열 생성

plt.hist(skills, 100, density=30)
plt.xlabel('True Skill')
plt.ylabel('Probability')
plt.title('Histogram of True Skill')
plt.axis([0, 3000, 0, 0.0015])
plt.grid(True)
plt.show()

# +
# 초기 사용자 정보 입력: id(1~10000), skill(위 정규분포 난수), mu(초기 rating값 = 1500), rd(초기 rd값 = 350), v(초기 volatility값 = 0.06)

raw_users = np.array([
    [
        _id,
        max(10, int(skill)),
        mu,
        350.0,
        0.06
    ]
    for _id, skill in zip(range(1, size+1), skills) 
])

# raw_usres의 skill 배열에 대한 0, 35 , 70, 92, 100 백분위수 배열

point_5=np.percentile(raw_users[:,1],q=[0,35,70,92,100])  # 69, 1351, 1706, 2060, 2985

for i in range(size):
    if raw_users[i,1]<point_5[1]:
        raw_users[i,2]=1000  #실력이 35%이하면 rating = 1000 (아이언, 브론즈)
    elif raw_users[i,1]<point_5[2]:
        raw_users[i,2]=1500  #70%이하면 rating = 1500 (실버)
    elif raw_users[i,1]<point_5[3]:
        raw_users[i,2]=2000  #92%이하면 rating = 2000 (골드)
    else :
        raw_users[i,2]=2500  #상위 8%이내 rating = 2500 (플레~)

raw_users[:6] #Users의 id, skill, rating, RD, sigma 세팅


# -

def init(users): #user DB 초기화
    c.execute("DELETE FROM users")
    conn.commit()
    
    c.executemany("INSERT INTO users (id, skill, rating, rd, win, lose, draw, sigma) VALUES (?, ?, ?, ?, 0, 0, 0, ?)", users)
    conn.commit()


def enqueue(size=3000): #users DB에서 3000명 랜덤으로 뽑아 enqueue
    c.execute("DELETE FROM queue")
    conn.commit()
    
    c.execute("SELECT id, skill, rating, rd, win, lose, draw, sigma FROM users ORDER BY rating")
    users = c.fetchall()
    c.executemany("INSERT INTO queue (id, skill, rating, rd, win, lose, draw, sigma) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", random.sample(users, size))
    conn.commit()


def matchmake(match_size=2): #rating +-2*rd인 사용자와 매칭시켜줌 [(,),(,),..] 튜플의 리스트 반환
    matches = []
    run_count = 0
    while True:
        run_count += 1
        c.execute("SELECT id, skill, rating, rd, win, lose, draw, sigma FROM queue ORDER BY rating LIMIT 1")
        entry = c.fetchone()
        if not entry:
            break
        else:
            c.execute("DELETE FROM queue WHERE id = ?", (entry[0],))
            conn.commit()
        
        #if run_count % 1000 == 0:
            #print('run: ', run_count)

        match_found = False
        candies = [entry]
        
        c.execute("SELECT id, skill, rating, rd, win, lose, draw, sigma FROM queue WHERE rating <= ? AND rating >= ? ORDER BY rating LIMIT 50", (entry[2]+2*entry[3], entry[2]-2*entry[3]))
        entries = list(c.fetchall())
        random.shuffle(entries)
        
        for idx, candi in enumerate(entries):
            if candi[2] - 2*candi[3] < entry[2] < candi[2] + 2*candi[3]:
                candies.append(candi)
                
            if len(candies) == match_size:
                matches.append(candies)
                match_found = True
                for el in candies:
                    c.execute("DELETE FROM queue WHERE id = ?", (el[0],))
                    conn.commit()
                break
                
        if not match_found:
            print('not found match')

    c.execute("DELETE FROM queue")
    conn.commit()

    return matches


def display():
    c.execute('SELECT id, skill, rating, win, lose, draw, sigma FROM users ORDER BY rating')
    res = np.array(sorted(list(c.fetchall()), key=lambda x: x[0]))
    pprint(res[-10:])
    
    user_ids = res[:, 0]
    user_skills = res[:, 1]
    user_ratings = res[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line = [min(user_skills), max(user_skills)]
    ax.plot(line, line, '--')
    ax.set_ylim([min(user_ratings)-100, max(user_ratings)+100])
    fig.tight_layout()

    plt.scatter(x=user_skills, y=user_ratings, color='g')
    plt.ylabel('rating')
    plt.xlabel('strength')
    plt.show()


def get_Sa(userA, userB): #대결 방식은 Elo rating에 따른 확률 싸움으로 정함, 1100 vs 1500 이면 400차이가 나니까 1100이 약 9%확률로 이김
                          # userA의 skill은 1500, userB의 skill은 1100이라 해보자.
    momentum = 1 #단판 승부
    '''
    probA = np.random.randint(low=userA[1]-400, high=userA[1], size=momentum)
    probB = np.random.randint(low=userB[1]-400, high=userB[1], size=momentum)
    '''
    probA = np.random.randint(low=1,high=100,size=momentum) #우선 1~100 이내 정수를 랜덤하게 뽑음
    probB = []
    for i in range(momentum):
        probB.append(int(1/(1+10**((userA[1]-userB[1])/400))*100)) #Elo rating에 따른 B의 예상승률 * 100 = 9 
    
    W = 0
    D = 0
    L = 0
    for pa, pb in zip(probA, probB): 
        if pa > pb: #여기서 probA가 9보다 크면 A가 이김(결국 91%확률로 A가 이김)
            W += 1
        elif pa == pb: 
            D += 1
        else:
            L += 1 #probA가 9보다 작으면 B가 이김(9%확률로 B가 이김)
    
    if W > L:
        return 1
    elif W == L:
        return 0.5
    else:
        return 0


# <h1>Glicko-2 volatility</h1>

# +
# Glicko-2를 위한 상수 선언 
TAU=0.5 
eps = 0.000001 
SCALE = 173.7178 

def scale_down(user):
    return [(user[2]-1500)/SCALE, user[3]/SCALE]

def scale_up(mu, RD):
    return [SCALE*mu+1500, SCALE*RD]

def g(RD):
    return 1 / math.sqrt(1 + 3*math.pow(RD, 2) / math.pi**2)

def E(mu, muj, RDj):
    return 1 / (1 + math.exp(-1*g(RDj)*(mu-muj)))

def compute_v(mu, muj, RDj):
    expected = E(mu, muj, RDj)
    return 1 / (g(RDj)**2 * expected * (1 - expected))


# +
def f(x, delta, RD, v, a):
    ex = math.exp(x)
    num1 = ex * (delta**2 - RD**2 - v - ex)
    denom1 = 2 * ((RD**2 + v + ex)**2)
    return  (num1 / denom1) - ((x - a) / (TAU**2))

def compute_delta(mu,muj,RDj,S,v):
    return v*g(RDj)*(S-E(mu,muj,RDj))

def compute_SIGMA(delta,RD,SIGMA,v): #Illinois algorithm (regula falsi)
    a = math.log(SIGMA**2)
    A = a
    B = None
    if (delta ** 2)  > ((RD**2) + v):
        B = math.log(delta**2 - RD**2 - v)
    else:        
        k = 1
        while f(a - k * TAU, RD, delta, v, a) < 0:
            k = k + 1
        B = a - k * TAU

    fA = f(A, delta, RD, v, a) #negative value
    fB = f(B, delta, RD, v, a) #positive value
        
    while math.fabs(B - A) > eps:
        C = A + ((A - B) * fA)/(fB - fA)
        fC = f(C, delta, RD, v, a)
        if fC * fB < 0:
            A = B
            fA = fB
        else:
            fA = fA/2.0
        B = C
        fB = fC
        
    return math.exp(A / 2)


# -

compute_SIGMA(-0.4834,1.1513,0.06,1.7785) #Example

pprint(scale_down([1, 1500, 1500, 200]))
pprint(scale_down([2, 1400, 1400, 30]))
pprint(scale_up(0, 1.1512924985234674))
pprint(scale_up(-0.5756462492617337, 0.1726938747785201))


# +
def compute_RD_star(RD,SIGMA):
    return math.sqrt(RD**2 + SIGMA**2)

def compute_RD_dash(RD, v, SIGMA):
    RD_star = math.sqrt(RD**2 + SIGMA**2)
    return 1 / math.sqrt(1/(RD_star**2) + 1/v)

def compute_mu_dash(mu, RD_dash, muj, RDj, S):
    return mu + RD_dash**2 * g(RDj) * (S - E(mu, muj, RDj))


# -

def match_glicko(userA, userB): #매칭된 userA와 userB를 서로 대결하게 시키고 결과값, 바뀐 rating, RD, sigma(v)를 반환
    S1 = get_Sa(userA, userB) #대결후 결과값 대입 (S1=1이면 A 승리)
    S2 = 1 - S1
    
    #사용자 전적 레코드 DB에 기록 
    c.execute("INSERT INTO user"+str(userA[0])+"_record (id, opp_id, opp_rating, opp_rd, result) VALUES (?, ?, ?, ?, ?)", (userA[0],userB[0],userB[2],userB[3],S1))
    conn.commit()
    c.execute("INSERT INTO user"+str(userB[0])+"_record (id, opp_id, opp_rating, opp_rd, result) VALUES (?, ?, ?, ?, ?)", (userB[0],userA[0],userA[2],userA[3],S2))
    conn.commit()
    
    mu1, RD1 = scale_down(userA)
    mu2, RD2 = scale_down(userB)
    SIGMA_A = userA[7]
    SIGMA_B = userB[7]
    
    v1 = compute_v(mu1, mu2, RD2)
    delta1 = compute_delta(mu1, mu2, RD2, S1, v1)
    SIGMA1 = compute_SIGMA(delta1, RD1, SIGMA_A, v1)
    RD_dash1 = compute_RD_dash(RD1, v1, SIGMA1)
    mu_dash1 = compute_mu_dash(mu1, RD_dash1, mu2, RD2, S1)
    
    v2 = compute_v(mu2, mu1, RD1)
    delta2 = compute_delta(mu2, mu1, RD1, S2, v2)
    SIGMA2 = compute_SIGMA(delta2, RD2, SIGMA_B, v2)
    RD_dash2 = compute_RD_dash(RD2, v2, SIGMA2)
    mu_dash2 = compute_mu_dash(mu2, RD_dash2, mu1, RD1, S2)
    
    return ([S1,SIGMA1] + scale_up(mu_dash1, RD_dash1)), ([S2,SIGMA2] + (scale_up(mu_dash2, RD_dash2)))


def match_and_update_ratings_glicko(rounds): #round마다 큐를 생성하고 matchmaking하고 대결시킨 후 결과를 받아서 users DB에 기록
    for i in range(rounds):
        print('epoch: ', i)
        enqueue()
        matches = matchmake()
        for userA, userB in matches:
            R1, R2 = match_glicko(userA, userB)
            
            (S1, sigma1, mu1, RD1) = R1
            (S2, sigma2, mu2, RD2) = R2

            aWin = userA[4] + 1 if S1 == 1 else userA[4]
            aLose = userA[5] + 1 if S1 == 0 else userA[5]
            aDraw = userA[6] + 1 if S1 == 0.5 else userA[6]

            bWin = userB[4] + 1 if S2 == 1 else userB[4]
            bLose = userB[5] + 1 if S2 == 0 else userB[5]
            bDraw = userB[6] + 1 if S2 == 0.5 else userB[6]

            c.execute("UPDATE users SET rating=?, RD=?, win=?, lose=?, draw=?, sigma=? WHERE id=?", (int(mu1), RD1, aWin, aLose, aDraw, round(sigma1,6), userA[0]))
            conn.commit()
            c.execute("UPDATE users SET rating=?, RD=?, win=?, lose=?, draw=?, sigma=? WHERE id=?", (int(mu2), RD2, bWin, bLose, bDraw, round(sigma2,6), userB[0]))
            conn.commit()


init(raw_users)

rounds = 30
match_and_update_ratings_glicko(rounds)

display()

rounds = 60
match_and_update_ratings_glicko(rounds)

display()

rounds = 180
match_and_update_ratings_glicko(rounds)

display()

c.execute("select * from users where id=100")
record_100=c.fetchall()
record_100

c.execute("select * from user100_record")
record_100=c.fetchall()
record_100


