#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_heap_caps.h"
#include "esp_pm.h"           // Power management / CPU freq
#include "esp_timer.h"

// ================== ESP-NN HARDWARE ACCELERATION ==================
// Tells Edge Impulse to use Espressif's optimised neural-net kernels
// (SIMD dot-product, optimised conv, depthwise-conv, etc.)
// Must be defined BEFORE any EI headers are included.
#define EI_CLASSIFIER_USE_FULL_TFLITE 0   // use EON compiler path, not full TFLite
#ifndef CONFIG_IDF_TARGET_ESP32S3
#define CONFIG_IDF_TARGET_ESP32S3         // ensure ESP-NN selects S3 SIMD paths
#endif

#include "ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/image/image.hpp"

// ================== NETWORK CREDENTIALS ==================
#define WIFI_SSID "Susmoy420"
#define WIFI_PASS "9876543210"

static const char *TAG = "AEGIS_YOLO_CORE";

// Camera Pin Mappings (ESP32-S3-EYE)
#define CAM_PIN_PWDN  -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK  15
#define CAM_PIN_SIOD  4
#define CAM_PIN_SIOC  5
#define CAM_PIN_D7    16
#define CAM_PIN_D6    17
#define CAM_PIN_D5    18
#define CAM_PIN_D4    12
#define CAM_PIN_D3    10
#define CAM_PIN_D2    8
#define CAM_PIN_D1    9
#define CAM_PIN_D0    11
#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF  7
#define CAM_PIN_PCLK  13

#define CAM_WIDTH  320
#define CAM_HEIGHT 240
#define MAX_DETECTIONS 16
#define DETECTION_THRESHOLD 0.0f   

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ================== SHARED DETECTION RESULTS ==================
typedef struct {
    char     label[32];
    float    score;
    uint32_t x, y, w, h;   // model-input pixel space (160x160)
} detection_t;

static detection_t       s_detections[MAX_DETECTIONS];
static int               s_detection_count = 0;
static int               s_last_inference_ms = 0;
static SemaphoreHandle_t s_det_mutex = NULL;

// ================== PERSISTENT BUFFERS ==================
static uint8_t *s_model_rgb_buffer  = NULL;
static uint8_t *s_decode_rgb_buffer = NULL;

// ================== EI DATA CALLBACK ==================
static int get_model_tensor_data(size_t offset, size_t length, float *out_ptr) {
    if (!s_model_rgb_buffer) return -1;
    size_t pixel_ix = offset * 3;
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)(
            ((uint32_t)s_model_rgb_buffer[pixel_ix]     << 16) +
            ((uint32_t)s_model_rgb_buffer[pixel_ix + 1] <<  8) +
             (uint32_t)s_model_rgb_buffer[pixel_ix + 2]
        );
        pixel_ix += 3;
    }
    return 0;
}

// ================== HTML UI ==================
static const char* INDEX_HTML = R"rawhtml(<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AEGIS Vision</title>
<style>
  body { margin:0; background:#111; display:flex; flex-direction:column;
         align-items:center; justify-content:center; min-height:100vh; font-family:monospace; }
  h1   { color:#0f0; margin:12px 0 8px; font-size:1.1em; letter-spacing:2px; }
  #wrap { position:relative; display:inline-block; }
  #stream { display:block; width:640px; height:480px; image-rendering:pixelated; }
  #overlay { position:absolute; top:0; left:0; width:640px; height:480px; pointer-events:none; }
  #status { color:#0f0; font-size:0.75em; margin-top:6px; }
</style>
</head>
<body>
<h1>&#9632; AEGIS YOLO VISION</h1>
<div id="wrap">
  <img id="stream" src="/stream">
  <canvas id="overlay" width="640" height="480"></canvas>
</div>
<div id="status">Waiting for detections...</div>
<script>
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');
const status = document.getElementById('status');
const MODEL_W = 160, MODEL_H = 160;
const DISP_W  = 640, DISP_H  = 480;
const COLORS  = ['#00ff00','#ff3300','#00aaff','#ffaa00','#ff00ff','#00ffcc'];

function colorFor(label) {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) & 0xffff;
  return COLORS[h % COLORS.length];
}

async function fetchDetections() {
  try {
    const r = await fetch('/detections');
    if (!r.ok) return;
    const data = await r.json();
    ctx.clearRect(0, 0, DISP_W, DISP_H);
    if (!data.detections || data.detections.length === 0) {
      status.textContent = `No detections | ${data.ms}ms inference`;
      return;
    }
    const scaleX = DISP_W / MODEL_W;
    const scaleY = DISP_H / MODEL_H;
    data.detections.forEach(d => {
      const x = d.x * scaleX, y = d.y * scaleY;
      const w = d.w * scaleX, h = d.h * scaleY;
      const col = colorFor(d.label);
      ctx.strokeStyle = col; ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const label = `${d.label} ${(d.score*100).toFixed(0)}%`;
      ctx.font = 'bold 13px monospace';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = col;
      ctx.fillRect(x, y - 18, tw + 6, 18);
      ctx.fillStyle = '#000';
      ctx.fillText(label, x + 3, y - 4);
    });
    status.textContent = `${data.detections.length} detection(s) | ${data.ms}ms inference`;
  } catch(e) { status.textContent = 'Connection error'; }
}
setInterval(fetchDetections, 200);
</script>
</body>
</html>)rawhtml";

// ================== HTTP HANDLERS ==================
esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

esp_err_t detections_handler(httpd_req_t *req) {
    char json[1024];
    int  pos = 0;
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    xSemaphoreTake(s_det_mutex, portMAX_DELAY);
    pos += snprintf(json + pos, sizeof(json) - pos,
                    "{\"count\":%d,\"ms\":%d,\"detections\":[",
                    s_detection_count, s_last_inference_ms);
    for (int i = 0; i < s_detection_count && pos < (int)sizeof(json) - 80; i++) {
        detection_t *d = &s_detections[i];
        pos += snprintf(json + pos, sizeof(json) - pos,
                        "%s{\"label\":\"%s\",\"score\":%.3f,"
                        "\"x\":%lu,\"y\":%lu,\"w\":%lu,\"h\":%lu}",
                        (i > 0 ? "," : ""),
                        d->label, d->score,
                        (unsigned long)d->x, (unsigned long)d->y,
                        (unsigned long)d->w, (unsigned long)d->h);
    }
    pos += snprintf(json + pos, sizeof(json) - pos, "]}");
    xSemaphoreGive(s_det_mutex);

    return httpd_resp_send(req, json, pos);
}

esp_err_t stream_handler(httpd_req_t *req) {
    esp_err_t res = ESP_OK;
    char part_buf[64];
    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;
    ESP_LOGI(TAG, "Stream client connected");
    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) { res = ESP_FAIL; break; }
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        esp_camera_fb_return(fb);
        vTaskDelay(pdMS_TO_TICKS(50));
        if (res != ESP_OK) break;
    }
    return res;
}

// Dedicated stream task pinned to Core 0 alongside Wi-Fi
// Keeps Core 1 free for uninterrupted inference
void stream_server_task(void *pvParameters) {
    httpd_config_t config   = HTTPD_DEFAULT_CONFIG();
    config.server_port      = 80;
    config.max_uri_handlers = 8;
    // Run the HTTP server's internal socket polling on Core 0
    config.core_id          = 0;
    httpd_handle_t server   = NULL;

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t uris[] = {
            { .uri = "/",           .method = HTTP_GET, .handler = index_handler,      .user_ctx = NULL },
            { .uri = "/stream",     .method = HTTP_GET, .handler = stream_handler,     .user_ctx = NULL },
            { .uri = "/detections", .method = HTTP_GET, .handler = detections_handler, .user_ctx = NULL },
        };
        for (int i = 0; i < 3; i++)
            httpd_register_uri_handler(server, &uris[i]);
        ESP_LOGI(TAG, "Server up — open http://192.168.0.236/ in your browser");
    }
    // Task stays alive to keep the server running
    vTaskDelete(NULL);
}

// ================== WIFI ==================
static void wifi_event_handler(void *arg, esp_event_base_t base,
                                int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "IP: " IPSTR, IP2STR(&ev->ip_info.ip));
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    }
}

static void wifi_init(void) {
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,    wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT,   IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL, NULL);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    wifi_config_t wifi_config = {};
    strcpy((char *)wifi_config.sta.ssid,     WIFI_SSID);
    strcpy((char *)wifi_config.sta.password, WIFI_PASS);
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    ESP_LOGI(TAG, "Wi-Fi connecting...");
    vTaskDelay(pdMS_TO_TICKS(4000));
}

// ================== CAMERA ==================
static esp_err_t init_hardware_camera(void) {
    camera_config_t config = {
        .pin_pwdn  = CAM_PIN_PWDN,  .pin_reset = CAM_PIN_RESET,
        .pin_xclk  = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD, .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7, .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5, .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3, .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1, .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC, .pin_href = CAM_PIN_HREF, .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = 20000000,
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size   = FRAMESIZE_QVGA,
        .jpeg_quality = 12,
        .fb_count     = 2,
        .fb_location  = CAMERA_FB_IN_PSRAM,
        .grab_mode    = CAMERA_GRAB_LATEST,
    };
    return esp_camera_init(&config);
}

// ================== AI CORE (Core 1, high priority) ==================
void inference_worker_task(void *pvParameters) {
    ESP_LOGI(TAG, "AI Core (Core 1): allocating PSRAM buffers...");

    size_t model_rgb_size  = (size_t)EI_CLASSIFIER_INPUT_WIDTH *
                             (size_t)EI_CLASSIFIER_INPUT_HEIGHT * 3;
    s_model_rgb_buffer = (uint8_t *)heap_caps_malloc(model_rgb_size, MALLOC_CAP_SPIRAM);
    if (!s_model_rgb_buffer) {
        ESP_LOGE(TAG, "FATAL: model buffer alloc failed"); vTaskDelete(NULL); return;
    }

    size_t decode_rgb_size = (size_t)CAM_WIDTH * (size_t)CAM_HEIGHT * 3;
    s_decode_rgb_buffer = (uint8_t *)heap_caps_malloc(decode_rgb_size, MALLOC_CAP_SPIRAM);
    if (!s_decode_rgb_buffer) {
        ESP_LOGE(TAG, "FATAL: decode buffer alloc failed"); vTaskDelete(NULL); return;
    }

    ESP_LOGI(TAG, "Buffers OK — model:%uB decode:%uB",
             (unsigned)model_rgb_size, (unsigned)decode_rgb_size);

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data     = &get_model_tensor_data;

    // Sensor warmup
    for (int i = 0; i < 5; i++) {
        camera_fb_t *f = esp_camera_fb_get();
        if (f) esp_camera_fb_return(f);
        vTaskDelay(pdMS_TO_TICKS(120));
    }
    ESP_LOGI(TAG, "Warmup done. Inference loop on Core 1.");

    while (1) {
        // 1. Grab JPEG frame
        camera_fb_t *frame = esp_camera_fb_get();
        if (!frame) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }

        // 2. Decode JPEG → BGR888 into pre-allocated PSRAM buffer
        bool ok = fmt2rgb888(frame->buf, frame->len, frame->format, s_decode_rgb_buffer);
        esp_camera_fb_return(frame);   // return immediately — Core 0 stream needs it
        if (!ok) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }

        // 3. BGR → RGB (OV2640 JPEG decoder outputs BGR, EI model expects RGB)
        {
            size_t px = (size_t)CAM_WIDTH * (size_t)CAM_HEIGHT;
            uint8_t *p = s_decode_rgb_buffer;
            for (size_t i = 0; i < px; i++, p += 3) {
                uint8_t t = p[0]; p[0] = p[2]; p[2] = t;
            }
        }

        // 4. Crop + bilinear resize → model input size (160x160)
        ei::image::processing::crop_and_interpolate_rgb888(
            s_decode_rgb_buffer, CAM_WIDTH,                CAM_HEIGHT,
            s_model_rgb_buffer,  EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT
        );

        // 5. Run inference — ESP-NN SIMD kernels active on ESP32-S3
        int64_t t0 = esp_timer_get_time();
        ei_impulse_result_t result = {0};
        EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
        int infer_ms = (int)((esp_timer_get_time() - t0) / 1000);

        if (err != EI_IMPULSE_OK) {
            ESP_LOGE(TAG, "run_classifier error: %d", err);
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        ESP_LOGI(TAG, "DSP:%dms Infer:%dms (wall:%dms) boxes:%u",
                 result.timing.dsp, result.timing.classification, infer_ms,
                 (unsigned)result.bounding_boxes_count);

        // 6. Publish results (mutex-protected, non-blocking for inference loop)
        if (xSemaphoreTake(s_det_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            s_detection_count   = 0;
            s_last_inference_ms = infer_ms;

            for (size_t ix = 0; ix < result.bounding_boxes_count &&
                                 s_detection_count < MAX_DETECTIONS; ix++) {
                auto &bb = result.bounding_boxes[ix];
                if (bb.value < DETECTION_THRESHOLD) continue;

                detection_t *d = &s_detections[s_detection_count++];
                strncpy(d->label, bb.label, sizeof(d->label) - 1);
                d->label[sizeof(d->label) - 1] = '\0';
                d->score = bb.value;
                d->x = bb.x; d->y = bb.y;
                d->w = bb.width; d->h = bb.height;

                ESP_LOGW(TAG, "  DETECT %-12s %.1f%%  x:%lu y:%lu w:%lu h:%lu",
                         bb.label, bb.value * 100.0f,
                         (unsigned long)bb.x, (unsigned long)bb.y,
                         (unsigned long)bb.width, (unsigned long)bb.height);
            }
            xSemaphoreGive(s_det_mutex);
        }

        if (s_detection_count == 0)
            ESP_LOGI(TAG, "  (no detections above threshold)");

        // No vTaskDelay here — run as fast as the model allows on Core 1
        // Core 0 handles Wi-Fi + stream independently
    }
}

// ================== MAIN ==================
extern "C" void app_main(void) {
    nvs_flash_init();

    // ── Lock CPU to 240 MHz ──────────────────────────────────────────────
    // Default is 160 MHz. 240 MHz gives ~33% more inference throughput.
    // esp_pm_configure locks both cores to max freq (no dynamic scaling).
    esp_pm_config_t pm_config = {
        .max_freq_mhz = 240,
        .min_freq_mhz = 240,   // disable power scaling — we want max perf
        .light_sleep_enable = false,
    };
    esp_err_t pm_err = esp_pm_configure(&pm_config);
    if (pm_err == ESP_OK)
        ESP_LOGI(TAG, "CPU locked at 240 MHz");
    else
        ESP_LOGW(TAG, "CPU freq lock failed (%d) — running at default", pm_err);

    s_det_mutex = xSemaphoreCreateMutex();

    wifi_init();
    init_hardware_camera();

    // ── Core 0: Wi-Fi + HTTP stream + detection API ──────────────────────
    // Pinned to Core 0 alongside the Wi-Fi/LwIP stack.
    // Priority 4 (one below inference) so inference never gets starved.
    xTaskCreatePinnedToCore(
        stream_server_task, "stream_server",
        8192, NULL, 4, NULL, 0
    );

    // ── Core 1: inference only — nothing else runs here ──────────────────
    // Priority 5, 40KB stack for Edge Impulse + TFLite arena.
    // With ESP-NN enabled and 240 MHz, expect ~2-4x speedup vs baseline.
    xTaskCreatePinnedToCore(
        inference_worker_task, "vision_core",
        40960, NULL, 5, NULL, 1
    );
}