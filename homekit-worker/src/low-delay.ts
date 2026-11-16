/** Bounded proof from FFmpeg framehash packets and HEVC trace_headers.
 * Intel may encode GPB slices as B pictures even with GopRefDist=1. Reject
 * temporal reordering and future references, rather than that picture label.
 * Neither packet contents nor diagnostic lines are retained.
 */
export class LowDelayProof {
  private pending = { packets: '', headers: '' };
  packets = 0;
  reorderHeaders = 0;
  idrs = 0;
  cra = 0;
  invalid = false;
  private previous = -Infinity;
  push(channel: 'packets' | 'headers', chunk: Buffer): void {
    this.pending[channel] += chunk.toString('utf8');
    for (let end; (end = this.pending[channel].indexOf('\n')) >= 0;) {
      const line = this.pending[channel].slice(0, end);
      this.pending[channel] = this.pending[channel].slice(end + 1);
      if (channel === 'packets') {
        const match = /^\s*0,\s*(-?\d+),\s*(-?\d+),/.exec(line);
        if (!match) continue;
        const dts = Number(match[1]), pts = Number(match[2]);
        if (dts !== pts || pts <= this.previous) this.invalid = true;
        this.previous = pts; this.packets++;
      } else {
        const reorder = /(?:vps|sps)_max_num_reorder_pics\[\d+\].*=\s*(\d+)/.exec(line);
        if (reorder) { this.reorderHeaders++; if (Number(reorder[1]) !== 0) this.invalid = true; }
        const future = /\bnum_positive_pics\s+.*=\s*(\d+)/.exec(line);
        if (future && Number(future[1]) !== 0) this.invalid = true;
        const type = /\bnal_unit_type\s+.*=\s*(\d+)/.exec(line);
        if (type) {
          if ([19, 20].includes(Number(type[1]))) this.idrs++;
          if (Number(type[1]) === 21) this.cra++;
        }
      }
    }
    if (this.pending[channel].length > 8192) { this.pending[channel] = ''; this.invalid = true; }
  }
  valid(frames: number): boolean { return !this.invalid && this.packets === frames && this.reorderHeaders > 0 && this.idrs >= 2 && this.cra === 0; }
}
