// Downsamples mic audio to 16 kHz mono PCM16 and posts ~40 ms frames to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.frame = new Int16Array(640); // 40 ms at 16 kHz
    this.filled = 0;
    this.pos = 0; // fractional read position; -1..0 means "between last block and this one"
    this.prev = 0; // last sample of the previous block
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    // Linear-interpolation resampler; good enough for speech recognition.
    while (this.pos < input.length - 1) {
      const i = Math.floor(this.pos);
      const f = this.pos - i;
      const a = i < 0 ? this.prev : input[i];
      const s = a * (1 - f) + input[i + 1] * f;
      this.frame[this.filled++] = Math.max(-1, Math.min(1, s)) * 0x7fff;
      if (this.filled === this.frame.length) {
        this.port.postMessage(this.frame.buffer.slice(0));
        this.filled = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= input.length;
    this.prev = input[input.length - 1];
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
