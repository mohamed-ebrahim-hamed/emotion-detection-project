// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // ===== DOM Elements =====
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('audioFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const recordBtn = document.getElementById('recordBtn');
    const stopRecordBtn = document.getElementById('stopRecordBtn');
    const resultContainer = document.getElementById('resultContainer');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const emotionDisplay = document.getElementById('emotionDisplay');
    const emotionEmoji = document.getElementById('emotionEmoji');
    const emotionText = document.getElementById('emotionText');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceFill = document.getElementById('confidenceFill');
    const probabilitiesList = document.getElementById('probabilitiesList');
    const audioPlayer = document.getElementById('audioPlayer');
    const waveform = document.getElementById('audioWaveform');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const fileSelected = document.getElementById('fileSelected');
    const previewSection = document.getElementById('previewSection');
    const actionsSection = document.getElementById('actionsSection');
    const recordingStatus = document.getElementById('recordingStatus');
    const analyzeAgainBtn = document.getElementById('analyzeAgainBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    // Text analysis elements
    const textInput = document.getElementById('textInput');
    const analyzeTextBtn = document.getElementById('analyzeTextBtn');
    const clearTextBtn = document.getElementById('clearTextBtn');
    const textResultContainer = document.getElementById('textResultContainer');
    const textEmotionsGrid = document.getElementById('textEmotionsGrid');
    
    let mediaRecorder;
    let audioChunks = [];
    let audioContext;
    let analyser;
    let animationId;
    let currentFile = null;
    
    // ===== Tab Navigation =====
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
    
    function switchTab(tabName) {
        tabBtns.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }
    
    // ===== Upload Area Events =====
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    fileInput.addEventListener('change', handleFileSelect);
    recordBtn.addEventListener('click', startRecording);
    stopRecordBtn.addEventListener('click', stopRecording);
    analyzeBtn.addEventListener('click', handleAnalyze);
    analyzeAgainBtn.addEventListener('click', handleAnalyzeAgain);
    clearBtn.addEventListener('click', handleClear);
    
    // Text analysis event listeners
    if (analyzeTextBtn) {
        analyzeTextBtn.addEventListener('click', handleTextAnalyze);
    }
    if (clearTextBtn) {
        clearTextBtn.addEventListener('click', () => {
            textInput.value = '';
            textResultContainer.style.display = 'none';
        });
    }
    
    // Drag and Drop Handlers
    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    }
    
    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('audio/')) {
            fileInput.files = files;
            handleFileSelect({ target: { files: files } });
        } else {
            showError('الرجاء اختيار ملف صوتي');
        }
    }
    
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            currentFile = file;
            updateFileInfo(file);
        }
    }
    
    function updateFileInfo(file) {
        // Update file info display
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
        fileSelected.style.display = 'block';
        
        // Show preview and actions sections
        previewSection.style.display = 'block';
        actionsSection.style.display = 'flex';
        
        // Hide results and errors
        resultContainer.style.display = 'none';
        errorMessage.style.display = 'none';
        
        // Preview audio
        const audioURL = URL.createObjectURL(file);
        audioPlayer.src = audioURL;
        
        // Create waveform visualization
        createWaveform(audioURL);
    }
    
    function handleAnalyzeAgain() {
        fileInput.value = '';
        currentFile = null;
        fileSelected.style.display = 'none';
        previewSection.style.display = 'none';
        actionsSection.style.display = 'none';
        resultContainer.style.display = 'none';
        switchTab('upload');
    }
    
    function handleClear() {
        handleAnalyzeAgain();
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Analyze Handler
    async function handleAnalyze() {
        const file = fileInput.files[0];
        if (!file) {
            showError('الرجاء اختيار ملف صوتي أولاً');
            return;
        }
        
        if (!file.type.startsWith('audio/')) {
            showError('الرجاء اختيار ملف صوتي (WAV, MP3, M4A, OGG, WEBM)');
            return;
        }
        
        await sendAudioFile(file);
    }
    
    // Recording Functions
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Try to use audio/webm;codecs=opus if supported, otherwise use default
            let options = { mimeType: 'audio/webm;codecs=opus' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'audio/webm' };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options = {}; // Use browser default
                }
            }
            
            mediaRecorder = new MediaRecorder(stream, options);
            audioChunks = [];
            
            // Setup audio visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            drawWaveform();
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                // Use the actual recorded MIME type
                const mimeType = mediaRecorder.mimeType || 'audio/webm';
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                
                // Use .webm extension for webm files, let server handle conversion
                const extension = mimeType.includes('webm') ? 'webm' : 'wav';
                const audioFile = new File([audioBlob], `recording.${extension}`, { type: mimeType });
                
                // Update file input
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(audioFile);
                fileInput.files = dataTransfer.files;
                
                // Store current file
                currentFile = audioFile;
                
                // Update UI to show file info and action buttons
                updateFileInfo(audioFile);
                
                // Stop visualization
                cancelAnimationFrame(animationId);
                
                // Hide recording status
                if (recordingStatus) {
                    recordingStatus.style.display = 'none';
                }
            };
            
            mediaRecorder.start();
            recordBtn.disabled = true;
            stopRecordBtn.disabled = false;
            
        } catch (err) {
            showError('Error accessing microphone: ' + err.message);
        }
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            recordBtn.disabled = false;
            stopRecordBtn.disabled = true;
        }
    }
    
    function drawWaveform() {
        if (!analyser) return;
        
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const canvas = waveform;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        function draw() {
            animationId = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);
            
            ctx.fillStyle = 'rgb(240, 240, 240)';
            ctx.fillRect(0, 0, width, height);
            
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#4361ee';
            ctx.beginPath();
            
            const sliceWidth = width * 1.0 / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * height / 2;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            ctx.lineTo(width, height / 2);
            ctx.stroke();
        }
        
        draw();
    }
    
    function createWaveform(audioURL) {
        const canvas = waveform;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Load audio and create visualization
        const audio = new Audio();
        audio.src = audioURL;
        
        audio.addEventListener('loadedmetadata', () => {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaElementSource(audio);
            analyser = audioContext.createAnalyser();
            
            source.connect(analyser);
            analyser.connect(audioContext.destination);
            
            analyser.fftSize = 256;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            function drawStaticWaveform() {
                analyser.getByteTimeDomainData(dataArray);
                
                ctx.fillStyle = 'rgb(240, 240, 240)';
                ctx.fillRect(0, 0, width, height);
                
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#4361ee';
                ctx.beginPath();
                
                const sliceWidth = width * 1.0 / bufferLength;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    const v = dataArray[i] / 128.0;
                    const y = v * height / 2;
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                    
                    x += sliceWidth;
                }
                
                ctx.lineTo(width, height / 2);
                ctx.stroke();
            }
            
            drawStaticWaveform();
        });
    }
    
    // API Communication
    async function sendAudioFile(file) {
        // Hide actions section and show loading
        actionsSection.style.display = 'none';
        loading.style.display = 'block';
        errorMessage.style.display = 'none';
        resultContainer.style.display = 'none';
        
        const formData = new FormData();
        formData.append('audio', file);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayResult(data);
            } else {
                showError(data.error || 'An error occurred during prediction');
            }
        } catch (error) {
            showError('Network error: ' + error.message);
        } finally {
            loading.style.display = 'none';
        }
    }
    
    function displayResult(data) {
        // Update emotion display
        emotionDisplay.style.backgroundColor = data.emotion_color;
        emotionEmoji.textContent = data.emotion_arabic.split(' ')[0];
        emotionText.textContent = data.emotion_arabic;
        confidenceText.textContent = `الثقة: ${data.confidence}%`;
        
        // Update confidence bar
        confidenceFill.style.width = '0%';
        setTimeout(() => {
            confidenceFill.style.width = `${data.confidence}%`;
        }, 100);
        
        // Update probabilities
        probabilitiesList.innerHTML = '';
        Object.entries(data.probabilities).forEach(([emotion, percent]) => {
            const color = getEmotionColor(emotion);
            const item = document.createElement('div');
            item.className = 'probability-item';
            item.innerHTML = `
                <span class="emotion-label" style="color: ${color}">${emotion}</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 0%; background: ${color}"></div>
                </div>
                <span class="prob-percent">0%</span>
            `;
            probabilitiesList.appendChild(item);
            
            // Animate progress bar
            setTimeout(() => {
                const fill = item.querySelector('.prob-fill');
                const percentSpan = item.querySelector('.prob-percent');
                fill.style.width = `${percent}%`;
                percentSpan.textContent = `${percent.toFixed(1)}%`;
            }, 100);
        });
        
        // Show result with animation
        resultContainer.style.display = 'block';
        
        // Add celebration effect for high confidence
        if (data.confidence > 85) {
            createConfetti();
        }
    }
    
    function getEmotionColor(emotion) {
        const colors = {
            'angry': '#FF6B6B',
            'disgust': '#8AC926',
            'fear': '#7209B7',
            'happy': '#FFD166',
            'sad': '#118AB2',
            'surprise': '#EF476F',
            'neutral': '#06D6A0'
        };
        return colors[emotion] || '#4361ee';
    }
    
    function showError(message) {
        if (errorText) {
            errorText.textContent = message;
        } else {
            errorMessage.textContent = message;
        }
        errorMessage.style.display = 'block';
        loading.style.display = 'none';
        
        // Show action buttons again if file is selected
        if (fileInput.files && fileInput.files[0]) {
            actionsSection.style.display = 'flex';
        }
        
        // Hide error after 5 seconds
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }
    
    // Confetti effect for high confidence predictions
    function createConfetti() {
        const confettiCount = 100;
        const confettiContainer = document.createElement('div');
        confettiContainer.style.position = 'fixed';
        confettiContainer.style.top = '0';
        confettiContainer.style.left = '0';
        confettiContainer.style.width = '100%';
        confettiContainer.style.height = '100%';
        confettiContainer.style.pointerEvents = 'none';
        confettiContainer.style.zIndex = '1000';
        
        document.body.appendChild(confettiContainer);
        
        for (let i = 0; i < confettiCount; i++) {
            const confetti = document.createElement('div');
            confetti.style.position = 'absolute';
            confetti.style.width = '10px';
            confetti.style.height = '10px';
            confetti.style.backgroundColor = getRandomColor();
            confetti.style.borderRadius = '50%';
            confetti.style.top = '0';
            confetti.style.left = `${Math.random() * 100}%`;
            
            confettiContainer.appendChild(confetti);
            
            // Animate confetti
            const animation = confetti.animate([
                { transform: 'translateY(0) rotate(0deg)', opacity: 1 },
                { transform: `translateY(${window.innerHeight}px) rotate(${360 * Math.random()}deg)`, opacity: 0 }
            ], {
                duration: 1000 + Math.random() * 2000,
                easing: 'cubic-bezier(0.215, 0.61, 0.355, 1)'
            });
            
            animation.onfinish = () => confetti.remove();
        }
        
        // Remove container after animation
        setTimeout(() => {
            confettiContainer.remove();
        }, 3000);
    }
    
    function getRandomColor() {
        const colors = ['#FF6B6B', '#4CC9F0', '#7209B7', '#FFD166', '#06D6A0', '#4361EE', '#EF476F'];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    // ===== Text Analysis Functions =====
    async function handleTextAnalyze() {
        const text = textInput.value.trim();
        
        if (!text) {
            showError('الرجاء إدخال نص للتحليل');
            return;
        }
        
        // Show loading
        loading.style.display = 'block';
        textResultContainer.style.display = 'none';
        errorMessage.style.display = 'none';
        
        try {
            const response = await fetch('/predict-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayTextResults(data);
            } else {
                showError(data.error || 'حدث خطأ أثناء تحليل النص');
            }
        } catch (error) {
            showError('خطأ في الاتصال بالخادم: ' + error.message);
        } finally {
            loading.style.display = 'none';
        }
    }
    
    function displayTextResults(data) {
        textEmotionsGrid.innerHTML = '';
        
        const emotions = data.detected_emotions || [];
        
        if (emotions.length === 0) {
            textEmotionsGrid.innerHTML = '<p style="text-align: center; color: #666;">لم يتم اكتشاف عواطف واضحة في النص</p>';
        } else {
            emotions.forEach((emotion, index) => {
                const card = document.createElement('div');
                card.className = 'text-emotion-card';
                if (index === 0) {
                    card.classList.add('primary');
                }
                
                card.innerHTML = `
                    <div class="text-emotion-emoji">${emotion.emoji}</div>
                    <div class="text-emotion-name">${emotion.emotion}</div>
                    <div class="text-emotion-name-ar">${emotion.emotion_arabic}</div>
                    <div class="text-emotion-probability">${emotion.probability.toFixed(1)}%</div>
                `;
                
                textEmotionsGrid.appendChild(card);
            });
        }
        
        textResultContainer.style.display = 'block';
        
        // Smooth scroll to results
        textResultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Initialize
    waveform.width = waveform.offsetWidth;
    waveform.height = waveform.offsetHeight;
    
    // Check for browser compatibility
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        recordBtn.style.display = 'none';
        stopRecordBtn.style.display = 'none';
        document.querySelector('.recorder p').textContent = 'Recording not supported in this browser';
    }
});