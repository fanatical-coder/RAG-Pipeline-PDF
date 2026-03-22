import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpEventType } from '@angular/common/http';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './home.component.html',
  styleUrl: './home.scss',
})
export class HomeComponent {
  query = '';
  loading = false;
  result: any = null;

  // Upload state
  uploading = false;
  uploadMessage = '';
  uploadSuccess = false;
  selectedFile: File | null = null;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef,
    private authService: AuthService
  ) { }

  getImageUrl(imagePath: string): string {
    const normalizedPath = imagePath.replace(/\\/g, '/');
    return `http://127.0.0.1:8000/image-file?path=${encodeURIComponent(normalizedPath)}`;
  }

  userPdfs: any[] = [];
  loadingPdfs = false;

  ngOnInit() {
    this.loadUserPdfs();
  }

  loadUserPdfs() {
    this.loadingPdfs = true;
    this.http.get<any>('http://127.0.0.1:8000/pdfs').subscribe({
      next: (res) => {
        this.userPdfs = res.pdfs;
        this.loadingPdfs = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.loadingPdfs = false;
      }
    });
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      this.selectedFile = file;
      this.uploadMessage = `Selected: ${file.name}`;
      this.uploadSuccess = false;
    } else {
      this.uploadMessage = 'Please select a valid PDF file.';
      this.selectedFile = null;
    }
  }

  uploadPdf() {
    if (!this.selectedFile) return;

    this.uploading = true;
    this.uploadMessage = 'Uploading...';
    this.uploadSuccess = false;

    const formData = new FormData();
    const filename = this.selectedFile.name;
    formData.append('file', this.selectedFile);

    this.http.post<any>('http://127.0.0.1:8000/upload', formData).subscribe({
      next: (res) => {
        this.uploadMessage = '⚙️ Processing your PDF... this may take a minute';
        this.selectedFile = null;
        this.cdr.detectChanges();
        this.pollStatus(filename); // 👈 start polling
      },
      error: (err) => {
        this.uploading = false;
        this.uploadSuccess = false;
        this.uploadMessage = `Error: ${err.error?.detail || 'Upload failed'}`;
        this.cdr.detectChanges();
      }
    });
  }

  pollStatus(filename: string) {
    const interval = setInterval(() => {
      this.http.get<any>(`http://127.0.0.1:8000/upload/status/${filename}`)
        .subscribe({
          next: (res) => {
            if (res.status === 'complete') {
              clearInterval(interval);
              this.uploading = false;
              this.uploadSuccess = true;
              this.uploadMessage = `✔ '${filename}' processed successfully! You can now search it.`;
              this.cdr.detectChanges();
            } else {
              // Still processing — update message with dots animation
              const dots = '.'.repeat((Date.now() / 500 % 4) | 0);
              this.uploadMessage = `⚙️ Processing your PDF${dots} this may take a minute`;
              this.cdr.detectChanges();
            }
          },
          error: () => clearInterval(interval)
        });
    }, 3000); // poll every 3 seconds
  }

  search() {
    this.loading = true;
    this.result = null;
    this.cdr.detectChanges();

    this.http
      .post<any>('http://127.0.0.1:8000/search', { query: this.query })
      .subscribe({
        next: (res) => {
          this.result = res;
          this.loading = false;
          this.cdr.detectChanges();
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

  logout() {
    this.authService.logout();
  }
}