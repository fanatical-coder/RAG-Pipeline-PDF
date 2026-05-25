import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LoadingBar } from './loading-bar';

describe('LoadingBar', () => {
  let component: LoadingBar;
  let fixture: ComponentFixture<LoadingBar>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoadingBar]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LoadingBar);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
