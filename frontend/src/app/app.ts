import { Component, inject, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, Router, NavigationStart, NavigationEnd, NavigationCancel, NavigationError } from '@angular/router';
import { LoadingBarComponent } from './components/loading-bar/loading-bar.component';
import { LoadingService } from './services/loading.service';
import { Auth } from '@angular/fire/auth';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, LoadingBarComponent, CommonModule],
  template: `
    <app-loading-bar></app-loading-bar>
    <div *ngIf="authReady; else authLoading">
      <router-outlet></router-outlet>
    </div>
    <ng-template #authLoading>
      <div class="auth-loader">
        <div class="auth-spinner"></div>
      </div>
    </ng-template>
  `,
  styles: [`
    .auth-loader {
      height: 100vh; display: flex;
      align-items: center; justify-content: center;
      background: #f5f2ec;
    }
    .auth-spinner {
      width: 40px; height: 40px;
      border: 3px solid #c8d9bc;
      border-top-color: #3b5c3a;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  `]
})
export class App implements OnInit {
  private router = inject(Router);
  private loadingService = inject(LoadingService);
  private auth = inject(Auth);
  private cdr = inject(ChangeDetectorRef); // 👈 add this
  authReady = false;

  ngOnInit() {
    setTimeout(() => {
      this.authReady = true;
      this.cdr.detectChanges(); // 👈 force update
    }, 3000);

    this.auth.onAuthStateChanged(() => {
      this.authReady = true;
      this.cdr.detectChanges(); // 👈 force update
    });

    this.router.events.subscribe(event => {
      if (event instanceof NavigationStart) {
        this.loadingService.show();
      } else if (
        event instanceof NavigationEnd ||
        event instanceof NavigationCancel ||
        event instanceof NavigationError
      ) {
        setTimeout(() => this.loadingService.hide(), 300);
      }
    });
  }
}
