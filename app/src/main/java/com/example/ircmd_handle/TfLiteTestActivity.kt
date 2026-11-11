package com.example.ircmd_handle

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*

class TfLiteTestActivity : AppCompatActivity() {
    
    private lateinit var interpreter: Interpreter
    private lateinit var originalImageView: ImageView
    private lateinit var enhancedImageView: ImageView
    private lateinit var statusTextView: TextView
    private lateinit var processButton: Button
    
    companion object {
        private const val TAG = "TfLiteTest"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tflite_test)
        
        initializeViews()
        initializeModel()
    }
    
    private fun initializeViews() {
        originalImageView = findViewById(R.id.originalImageView)
        enhancedImageView = findViewById(R.id.enhancedImageView)
        statusTextView = findViewById(R.id.statusTextView)
        processButton = findViewById(R.id.processButton)
        
        processButton.setOnClickListener {
            testSuperResolution()
        }
        
        statusTextView.text = "🔄 正在加载 FSRCNN 模型..."
    }
    
    private fun initializeModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Loading FSRCNN model...")
                val modelBuffer = loadModelFromAssets("models/fsrcnn_x2.tflite")
                
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    // GPU delegate will be added once model loading is confirmed
                    // addDelegate(GpuDelegate())
                }
                
                interpreter = Interpreter(modelBuffer, options)
                
                // Log model details
                val inputTensor = interpreter.getInputTensor(0)
                val outputTensor = interpreter.getOutputTensor(0)
                
                Log.d(TAG, "Model loaded successfully")
                Log.d(TAG, "Input shape: ${inputTensor.shape().contentToString()}")
                Log.d(TAG, "Output shape: ${outputTensor.shape().contentToString()}")
                Log.d(TAG, "Input type: ${inputTensor.dataType()}")
                Log.d(TAG, "Output type: ${outputTensor.dataType()}")
                
                withContext(Dispatchers.Main) {
                    statusTextView.text = """
                        ✅ FSRCNN 模型加载成功！
                        
                        📋 模型详情：
                        • 输入：${inputTensor.shape().contentToString()}
                        • 输出：${outputTensor.shape().contentToString()}
                        • 数据类型：${inputTensor.dataType()}
                        
                        🎯 已准备好用 256x192 热成像图像进行测试！
                        
                        点击按钮开始测试超分辨率。
                    """.trimIndent()
                    
                    processButton.isEnabled = true
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Model loading failed", e)
                withContext(Dispatchers.Main) {
                    statusTextView.text = """
                        ❌ 模型加载失败：${e.message}
                        
                        🔍 排查建议：
                        • 确认 fsrcnn_x2.tflite 位于 app/src/main/assets/models/
                        • 检查模型转换是否成功
                        • 确认已添加 TensorFlow Lite 依赖
                        
                        错误详情：${e.javaClass.simpleName}
                    """.trimIndent()
                }
            }
        }
    }
    
    private fun testSuperResolution() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                withContext(Dispatchers.Main) {
                    statusTextView.text = "🔄 正在处理热成像图像..."
                    processButton.isEnabled = false
                }
                
                Log.d(TAG, "Starting super resolution test...")
                
                // Create test thermal image (256x192 grayscale)
                val originalBitmap = createTestThermalImage()
                Log.d(TAG, "Created test image: ${originalBitmap.width}x${originalBitmap.height}")
                
                // Enhance with FSRCNN
                val startTime = System.currentTimeMillis()
                val enhancedBitmap = enhanceImage(originalBitmap)
                val processingTime = System.currentTimeMillis() - startTime
                
                Log.d(TAG, "Enhancement complete: ${enhancedBitmap.width}x${enhancedBitmap.height}")
                Log.d(TAG, "Processing time: ${processingTime}ms")
                
                withContext(Dispatchers.Main) {
                    // Display results
                    originalImageView.setImageBitmap(originalBitmap)
                    enhancedImageView.setImageBitmap(enhancedBitmap)
                    
                    val scaleFactor = enhancedBitmap.width.toFloat() / originalBitmap.width
                    
                    statusTextView.text = """
                        ✅ 超分辨率处理完成！
                        
                        📊 结果：
                        • 输入：${originalBitmap.width}x${originalBitmap.height} 像素
                        • 输出：${enhancedBitmap.width}x${enhancedBitmap.height} 像素
                        • 放大倍数：${String.format("%.1f", scaleFactor)}x
                        • 处理耗时：${processingTime}ms
                        
                        🎯 非常适合热成像图像！
                        ${if (processingTime < 30) "⚡ 可实时处理！" else "⏱️ 建议进一步优化"}
                        
                        点击按钮可用不同图像再次测试。
                    """.trimIndent()
                    
                    processButton.isEnabled = true
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Processing failed", e)
                withContext(Dispatchers.Main) {
                    statusTextView.text = """
                        ❌ 处理失败：${e.message}
                        
                        🔍 错误详情：
                        ${e.javaClass.simpleName}: ${e.localizedMessage}
                        
                        可能的原因：
                        • 模型输入/输出格式不匹配
                        • 内存不足
                        • 预处理不正确
                    """.trimIndent()
                    processButton.isEnabled = true
                }
            }
        }
    }
    
    private fun createTestThermalImage(): Bitmap {
        // Create a 256x192 test image simulating thermal data
        val bitmap = Bitmap.createBitmap(256, 192, Bitmap.Config.ARGB_8888)
        
        Log.d(TAG, "Creating test thermal image...")
        
        // Create thermal-like patterns with hot spots and gradients
        for (y in 0 until 192) {
            for (x in 0 until 256) {
                // Create multiple thermal patterns
                
                // Gradient background
                val gradient = ((x + y) * 128 / (256 + 192)).coerceIn(0, 128)
                
                // Hot spot 1 (circle)
                val dx1 = x - 64
                val dy1 = y - 48
                val dist1 = sqrt((dx1 * dx1 + dy1 * dy1).toFloat())
                val hotspot1 = (255 * exp(-dist1 / 20)).toInt()
                
                // Hot spot 2 (circle)
                val dx2 = x - 192
                val dy2 = y - 144
                val dist2 = sqrt((dx2 * dx2 + dy2 * dy2).toFloat())
                val hotspot2 = (200 * exp(-dist2 / 25)).toInt()
                
                // Cold spot (negative gaussian)
                val dx3 = x - 128
                val dy3 = y - 96
                val dist3 = sqrt((dx3 * dx3 + dy3 * dy3).toFloat())
                val coldspot = -(100 * exp(-dist3 / 30)).toInt()
                
                // Combine patterns
                val combined = gradient + hotspot1 + hotspot2 + coldspot
                
                // Add some noise
                val noise = ((Math.random() - 0.5) * 20).toInt()
                val finalValue = (combined + noise).coerceIn(0, 255)
                
                bitmap.setPixel(x, y, Color.rgb(finalValue, finalValue, finalValue))
            }
        }
        
        Log.d(TAG, "Test thermal image created with realistic patterns")
        return bitmap
    }
    
    private fun enhanceImage(inputBitmap: Bitmap): Bitmap {
        Log.d(TAG, "Starting image enhancement...")
        
        // Preprocess: Bitmap → Float Array
        val inputArray = preprocessBitmap(inputBitmap)
        Log.d(TAG, "Preprocessing complete")
        
        // Get output tensor details to determine output size
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        
        Log.d(TAG, "Expected output shape: ${outputShape.contentToString()}")
        
        // Prepare output buffer
        val outputArray = if (outputShape[1] > 0 && outputShape[2] > 0) {
            // Fixed output shape
            Array(outputShape[0]) { Array(outputShape[1]) { Array(outputShape[2]) { FloatArray(outputShape[3]) } } }
        } else {
            // Dynamic output shape - assume 2x scaling
            Array(1) { Array(384) { Array(512) { FloatArray(1) } } }
        }
        
        Log.d(TAG, "Running FSRCNN inference...")
        
        // Run FSRCNN inference
        interpreter.run(inputArray, outputArray)
        
        Log.d(TAG, "Inference complete, postprocessing...")
        
        // Postprocess: Float Array → Bitmap
        return postprocessToBitmap(outputArray)
    }
    
    private fun preprocessBitmap(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        Log.d(TAG, "Preprocessing bitmap: ${bitmap.width}x${bitmap.height}")
        
        val inputArray = Array(1) { Array(192) { Array(256) { FloatArray(1) } } }
        
        for (y in 0 until 192) {
            for (x in 0 until 256) {
                val pixel = bitmap.getPixel(x, y)
                val gray = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
                inputArray[0][y][x][0] = gray / 255.0f  // Normalize to [0,1]
            }
        }
        
        Log.d(TAG, "Preprocessing complete: ${inputArray.size}x${inputArray[0].size}x${inputArray[0][0].size}x${inputArray[0][0][0].size}")
        return inputArray
    }
    
    private fun postprocessToBitmap(output: Array<Array<Array<FloatArray>>>): Bitmap {
        val height = output[0].size
        val width = output[0][0].size
        
        Log.d(TAG, "Postprocessing output: ${width}x${height}")
        
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // Find min/max for proper normalization
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE
        
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = output[0][y][x][0]
                minVal = min(minVal, value)
                maxVal = max(maxVal, value)
            }
        }
        
        Log.d(TAG, "Output range: [$minVal, $maxVal]")
        
        // Convert to bitmap with proper scaling
        for (y in 0 until height) {
            for (x in 0 until width) {
                val normalizedValue = if (maxVal > minVal) {
                    (output[0][y][x][0] - minVal) / (maxVal - minVal)
                } else {
                    output[0][y][x][0]
                }
                
                val pixelValue = (normalizedValue * 255).toInt().coerceIn(0, 255)
                val pixel = Color.rgb(pixelValue, pixelValue, pixelValue)
                bitmap.setPixel(x, y, pixel)
            }
        }
        
        Log.d(TAG, "Postprocessing complete")
        return bitmap
    }
    
    private fun loadModelFromAssets(filename: String): ByteBuffer {
        Log.d(TAG, "Loading model from assets: $filename")
        
        val assetFileDescriptor = assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        
        Log.d(TAG, "Model file size: $declaredLength bytes")
        
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        if (::interpreter.isInitialized) {
            interpreter.close()
            Log.d(TAG, "TensorFlow Lite interpreter closed")
        }
    }
}