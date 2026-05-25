import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LoadingService } from '../../services/loading.service';

@Component({
  selector: 'app-loading-bar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="loading-bar" *ngIf="loadingService.isLoading()">
      <div class="loading-progress"></div>
    </div>
  `,
  styleUrl: './loading-bar.component.scss'
})
export class LoadingBarComponent {
  loadingService = inject(LoadingService);
}