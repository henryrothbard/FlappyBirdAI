const BIRD_IMG_SRC = 'https://raw.githubusercontent.com/sourabhv/FlapPyBird/master/assets/sprites/yellowbird-midflap.png';
const PIPE_IMG_SRC = 'https://raw.githubusercontent.com/sourabhv/FlapPyBird/master/assets/sprites/pipe-green.png';

const GRAVITY = 0.01;
const FLAP_STRENGTH = GRAVITY * 2;
const SPEED = 10;
const BIRD_RADIUS = 0.04;
const BIRD_OFFSET = 0.1;
const BIRD_VISUAL_SCALE = 1.1;
const BIRD_ALPHA = 1;
const PIPE_WIDTH = 0.09;
const PIPE_GAP = 0.3;
const PIPE_RATE = 500/SPEED;

const GENERATION_SIZE = 512;
const NUM_ELITES = 32;
const LAYER_SIZES = [2, 8, 8, 1];
const ACTIVATION_FN = (x) => Math.max(0, x); // ReLU
const eps = 0.1;
const alpha = 0.05;

const loadImg = (src) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            img.ratio = img.naturalHeight/img.naturalWidth;
            resolve(img);
        }
        img.onerror = reject;
        img.src = src;
    })
}

let birdImg = loadImg(BIRD_IMG_SRC);
let pipeImg = loadImg(PIPE_IMG_SRC);

const generationStat = document.getElementById("generationStat");
const highscoreStat = document.getElementById("highscoreStat");
const aliveStat = document.getElementById("aliveStat");
const scoreStat = document.getElementById("scoreStat");
const speedSlider = document.getElementById("speedSlider");
const speedValue = document.getElementById("speedValue");

const TARGET_FPS = 60;
const MIN_DELAY = 0;
const MAX_DELAY = 1000;

let speedDelay = 1000 / TARGET_FPS;

function updateSpeed() {
    let percent = parseInt(speedSlider.value);
    speedValue.textContent = percent + "%";

    if (percent === 0) {
        speedDelay = MAX_DELAY;
    } else if (percent === 100) {
        speedDelay = 1000 / TARGET_FPS;
    } else if (percent > 100) {
        speedDelay = 1000 / TARGET_FPS * (200 - percent) / 100;
        if (speedDelay < MIN_DELAY) speedDelay = MIN_DELAY;
    } else {
        speedDelay = 1000 / TARGET_FPS + (100 - percent) * (MAX_DELAY - 1000 / TARGET_FPS) / 100;
    }
}
if (speedSlider && speedValue) {
    updateSpeed();
    speedSlider.addEventListener("input", updateSpeed);
}



const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false;

const resizeCanvas = () => {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

const drawBird = (y) => {
    const width = 2 * BIRD_RADIUS * canvas.height * BIRD_VISUAL_SCALE;
    const height = width * birdImg.ratio;
    const X = BIRD_OFFSET * BIRD_VISUAL_SCALE * canvas.width;
    const Y = (1-y) * canvas.height - height/2;
    ctx.save();
    ctx.globalAlpha = BIRD_ALPHA;
    ctx.drawImage(birdImg, X, Y, width, height);
    ctx.restore();
}

const drawPipe = (x, y) => {
    const width = PIPE_WIDTH * canvas.width;
    const height = width * pipeImg.ratio;
    const gap = PIPE_GAP * canvas.height;
    const X = (x * (1-PIPE_WIDTH)) * canvas.width;
    const Y = ((1-y) * (1-(PIPE_GAP))) * canvas.height;
    ctx.save();
    ctx.drawImage(pipeImg, X, Y + gap, width, height)
    ctx.scale(1, -1)
    ctx.drawImage(pipeImg, X, height - Y, width, -height)
    ctx.restore();
}

const deepMap = (array, fn, indices = []) => {
    return array.map((item, idx) =>
        Array.isArray(item)
            ? deepMap(item, fn, [...indices, idx])
            : fn(item, [...indices, idx])
    );
}

function randn() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function directionalNoise(shape, eps) {
    let n = shape[0] * shape[1];
    let noise = Array(n).fill(0).map(() => Math.random() * 2 - 1);
    let norm = Math.sqrt(noise.reduce((s, v) => s + v*v, 0)) || 1e-8;
    noise = noise.map(v => v / norm * eps);
    let arr = [];
    for (let i = 0; i < shape[0]; i++) {
        arr.push(noise.slice(i*shape[1], (i+1)*shape[1]));
    }
    return arr;
}

class Model {
    constructor() {
        this.zero();
    }

    zero() {
        this.W = [];
        this.b = [];
        for (let layer = 1; layer < LAYER_SIZES.length; layer++) {
            this.W.push([]);
            this.b.push([]);
            for (let node = 0; node < LAYER_SIZES[layer]; node++) {
                this.W[layer-1].push([]);
                this.b[layer-1].push(0);
                for (let prev = 0; prev < LAYER_SIZES[layer-1]; prev++) {
                    this.W[layer-1][node].push(0);
                }
            }
        }
    }

    randomizeWeights(randomizer) {
        this.W = deepMap(this.W, (w, idx) => {
            let layer = idx[0];
            let fanIn = LAYER_SIZES[layer];
            let scale = Math.sqrt(2 / fanIn); // He init for ReLU
            return randomizer() * scale;
        });
    }
    

    randomizeBiases(randomizer) {
        this.b = deepMap(this.b, (_0, _1) => randomizer())
    }

    copyWithNoise(eps = 0.1) {
        if (Math.random() < 0.5) {
            const newModel = new Model();
            newModel.W = this.W.map((layer, l) => {
                let rows = layer.length, cols = layer[0].length;
                const dirNoise = directionalNoise([rows, cols], eps);
                return layer.map((nodeArr, i) =>
                    nodeArr.map((w, j) => w + dirNoise[i][j])
                );
            });
            newModel.b = this.b.map((layer, l) => {
                let len = layer.length;
                let biasDirNoise = directionalNoise([1, len], eps)[0];
                return layer.map((b, i) => b + biasDirNoise[i]);
            });
            return newModel;
        } else {
            const newModel = new Model();
            newModel.W = this.W.map(layer =>
                layer.map(nodeArr =>
                    nodeArr.map(w => w + randn() * eps)
                )
            );
            newModel.b = this.b.map(layer =>
                layer.map(b => b + randn() * eps)
            );
            return newModel;
        }
    }

    forward(x) {
        let z = [];
        for (let layer = 0; layer < this.W.length; layer++) {
            z = [];
            for (let node = 0; node < this.W[layer].length; node++) {
                let sum = this.b[layer][node];
                for (let prev = 0; prev < this.W[layer][node].length; prev++) {
                    sum += this.W[layer][node][prev] * x[prev];
                }
                z.push(sum);
            }

            if (layer < this.W.length - 1) {
                x = z.map(ACTIVATION_FN);
            }
        }
        return z;
    }
}

const combineModels = (models, scores, alpha) => {
    let totalScore = scores.reduce((a, b) => a + b, 0);

    function blendFn_W(_, indices) {
        let sum = 0;
        for (let m = 0; m < models.length; m++) {
            let value = models[m].W;
            for (const idx of indices) value = value[idx];
            sum += scores[m] * value;
        }
        return alpha * sum / totalScore;
    }

    function blendFn_b(_, indices) {
        let sum = 0;
        for (let m = 0; m < models.length; m++) {
            let value = models[m].b;
            for (const idx of indices) value = value[idx];
            sum += scores[m] * value;
        }
        return alpha * sum / totalScore;
    }

    let newModel = new Model();
    newModel.W = deepMap(models[0].W, blendFn_W);
    newModel.b = deepMap(models[0].b, blendFn_b);
    return newModel;
}

class Environment {
    constructor() {
        this.reset();
    }

    reset() {
        this.pipes = [];
        this.frame = 0;
        this.spawnPipe();
    }

    spawnPipe() {
        this.pipes.push({ x: 1, y: 0.2 + 0.6 * Math.random() });
    }

    step() {
        this.frame++;

        for (const pipe of this.pipes) {
            pipe.x -= SPEED / canvas.width;
        }

        this.pipes = this.pipes.filter(pipe => pipe.x + PIPE_WIDTH > 0);

        if (this.frame % PIPE_RATE === 0) this.spawnPipe();

        for (const pipe of this.pipes) {
            drawPipe(pipe.x, pipe.y);
        }
    }

    nextPipe() {
        let birdX = BIRD_OFFSET;
        for (const pipe of this.pipes) {
            if (pipe.x + PIPE_WIDTH > birdX) {
                return pipe;
            }
        }
        return this.pipes[0] || {x: 1, y: 0.5};
    }

    isColliding(bird) {
        if (bird.y < BIRD_RADIUS || bird.y > 1 - BIRD_RADIUS) return true;

        let birdX = BIRD_OFFSET;
        for (const pipe of this.pipes) {
            if (birdX + BIRD_RADIUS > pipe.x && birdX - BIRD_RADIUS < pipe.x + PIPE_WIDTH) {
                if (bird.y < pipe.y - PIPE_GAP / 2 || bird.y > pipe.y + PIPE_GAP / 2) {
                    return true;
                }
            }
        }
        return false;
    }
}

const environment = new Environment();

const createGeneration = (parentModel, elites=[]) => {
    const birds = elites.map(m => new Bird(m));
    while (birds.length < GENERATION_SIZE) {
        let child = parentModel.copyWithNoise(eps);
        birds.push(new Bird(child));
    }
    return birds;
};

const makePhi = (scores) => {
    let sorted = [...scores].sort((a, b) => a - b);
    let ranks = scores.map(s => sorted.indexOf(s));
    return (score, i) => -1 + 2 * (ranks[i] / (scores.length - 1));
};

const combineGeneration = (birds) => {
    const models = birds.map(b => b.model);
    const scores = birds.map(b => b.score);
    const phi = makePhi(scores);
    const weights = scores.map((s, i) => phi(s, i));

    let eliteModel = models[scores.indexOf(Math.max(...scores))];
    let blended = combineModels(models, weights, alpha);

    return { blended, eliteModel };
};

class Bird {
    constructor(model) {
        this.model = model;
        this.alive = true;
        this.y = 0.5;
        this.vy = 0;
        this.score = 0;
        this.reward = 0;
    }

    step() {
        if (!this.alive) return;
        this.vy -= GRAVITY;

        let pipe = environment.nextPipe();
        let dy = this.y - pipe.y;
        let input = [dy, this.vy];
        let [out] = this.model.forward(input);

        
        if (out > 0) this.vy += FLAP_STRENGTH;
        
        this.y += this.vy;
        drawBird(this.y);
        
        let dist = Math.abs(this.y - pipe.y);
        let proximityReward = 1 - dist / 0.5;
        this.reward += proximityReward;
        
        if (environment.isColliding(this)) {
            this.alive = false;
            this.score = environment.frame + this.reward * 10;
        }
        
    }
}

const stepGeneration = (birds, environment) => {
    environment.step();
    let alive = 0;
    for (const bird of birds) {
        if (bird.alive) {
            bird.step();
            alive++;
        }
    }
    aliveStat.innerHTML = `${alive}/${GENERATION_SIZE}`
    scoreStat.innerHTML = environment.frame/PIPE_RATE;
    return alive;
};

let highscore = 0;

const main = async () => {
    [birdImg, pipeImg] = await Promise.all([birdImg, pipeImg]);

    let parent = new Model();
    parent.randomizeWeights(() => (Math.random() * 2 - 1) * 0.5);
    parent.randomizeBiases(() => (Math.random() * 2 - 1) * 0.5);

    let generation = 0;
    let elites = [];
    while (true) {
        generationStat.innerHTML = generation;
        environment.reset();
        let birds = createGeneration(parent, elites);
    
        let alive = true;
        while (alive) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            alive = stepGeneration(birds, environment);
            await new Promise(r => setTimeout(r, speedDelay));
        }
    
        highscore = Math.max(environment.frame/PIPE_RATE, highscore);
        highscoreStat.innerHTML = highscore;
    
        const scores = birds.map(b => b.score);
        const models = birds.map(b => b.model);
        let eliteIndices = scores.map((s, i) => [s, i])
            .sort((a, b) => b[0] - a[0])
            .slice(0, NUM_ELITES)
            .map(x => x[1]);
        elites = eliteIndices.map(i => models[i]);
    
        let { blended } = combineGeneration(birds);
    
        parent = blended;
    
        generation++;
        await new Promise(r => setTimeout(r, 0));
    }
};

main();