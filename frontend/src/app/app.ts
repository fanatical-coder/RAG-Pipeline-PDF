import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule, RouterOutlet],
  template: `<router-outlet></router-outlet>`
})
export class App {
  query = '';
  loading = false;
  result: any = null;

  constructor(private http: HttpClient, private cdr: ChangeDetectorRef) {}

  getImageUrl(imagePath: string): string {
    const normalizedPath = imagePath.replace(/\\/g, '/');
    return `http://127.0.0.1:8000/image-file?path=${encodeURIComponent(normalizedPath)}`;
  }

  search() {
    this.loading = true;
    this.result = null;
    this.cdr.detectChanges();

    this.http
      .post<any>('http://127.0.0.1:8000/search', { query: this.query })
      .subscribe({
        next: (res) => {
          console.log('Response:', res);
          this.result = res;
          this.loading = false;
          this.cdr.detectChanges();  // force view update
        },
        error: (err) => {
          console.error(err);
          this.loading = false;
          this.cdr.detectChanges();
        },
        complete: () => {
          this.loading = false;
          this.cdr.detectChanges();
        }
      });
  }
}