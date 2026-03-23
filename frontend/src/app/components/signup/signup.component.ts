import { Component, inject } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-signup',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule, RouterLink],
  templateUrl: './signup.component.html',
  styleUrl: './signup.component.scss',
})
export class SignupComponent {
  private fb = inject(FormBuilder);
  private authService = inject(AuthService);

  signupForm: FormGroup = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    password: ['', [Validators.required, Validators.minLength(6)]],
    confirmPassword: ['', Validators.required]
  });

  errorMessage = '';
  loading = false;
  googleLoading = false;
  googleError = '';

  async onSubmit() {
    if (this.signupForm.invalid) return;
    const { password, confirmPassword } = this.signupForm.value;
    if (password !== confirmPassword) {
      this.errorMessage = 'Passwords do not match';
      return;
    }
    this.loading = true;
    this.errorMessage = '';
    try {
      const { email } = this.signupForm.value;
      await this.authService.signUp(email, password);
    } catch (error: any) {
      this.errorMessage = error.message;
    } finally {
      this.loading = false;
    }
  }

  async loginWithGoogle() {
    this.googleLoading = true;
    this.googleError = '';
    try {
      await this.authService.loginWithGoogle();
    } catch (error: any) {
      this.googleError = error.message;
    } finally {
      this.googleLoading = false;
    }
  }
}