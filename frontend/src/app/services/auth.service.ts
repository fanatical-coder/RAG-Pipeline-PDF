import { Injectable, inject } from '@angular/core';
import { Auth, signInWithEmailAndPassword, 
         createUserWithEmailAndPassword, 
         signOut,user,GoogleAuthProvider,
         signInWithPopup } from '@angular/fire/auth';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private auth = inject(Auth);
  private router = inject(Router);

  // Observable of current user (null if logged out)
  currentUser$ = user(this.auth);

  // Sign Up
  async signUp(email: string, password: string) {
    try {
      await createUserWithEmailAndPassword(this.auth, email, password);
      this.router.navigate(['/home']);
    } catch (error: any) {
      throw error;
    }
  }

  // Login
  async login(email: string, password: string) {
    try {
      await signInWithEmailAndPassword(this.auth, email, password);
      this.router.navigate(['/home']);
    } catch (error: any) {
      throw error;
    }
  }

    async loginWithGoogle() {
    try {
      const provider = new GoogleAuthProvider();
      await signInWithPopup(this.auth, provider);
      this.router.navigate(['/home']);
    } catch (error: any) {
      throw error;
    }
  }

  // Logout
  async logout() {
    await signOut(this.auth);
    this.router.navigate(['/login']);
  }

  // Get current user's ID token (for FastAPI calls)
  async getIdToken(): Promise<string | null> {
    const currentUser = this.auth.currentUser;
    if (currentUser) {
      return await currentUser.getIdToken();
    }
    return null;
  }
}